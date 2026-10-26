import ARKit
import CoreGraphics
import CoreVideo
import simd

enum TerrainKind: String {
    case unknown = "路况待确认"
    case flat = "平路"
    case uphill = "上坡"
    case downhill = "下坡"
    case stairsUp = "上楼梯"
    case stairsDown = "下楼梯"
    case obstacle = "前方障碍"
    case dropOff = "前方深度断层"
}

struct TerrainResult {
    let kind: TerrainKind
    let confidence: Float
    let debug: String
    let distance: Float?

    var announcement: String {
        switch kind {
        case .unknown: return "前方路况待确认，请减速"
        case .flat: return "前方平路"
        case .uphill: return "前方上坡"
        case .downhill: return "前方下坡，注意脚下"
        case .stairsUp: return "前方上楼梯，注意台阶"
        case .stairsDown: return "前方下楼梯，注意台阶"
        case .obstacle:
            guard let distance = distance else { return "注意，前方有障碍" }
            let rounded = min(3.0, max(0.5, (distance * 2).rounded() / 2))
            return "注意，前方约 \(String(format: "%.1f", rounded)) 米有障碍"
        case .dropOff: return "注意，前方深度断层，请停止前行"
        }
    }
}

/// Conservative, on-device terrain estimation from ARKit scene depth.
/// Confidence means geometric evidence strength, not measured classification accuracy.
final class DepthTerrainAnalyzer {
    private struct Sample {
        let forward: Float
        let height: Float
        let depth: Float
    }

    private struct Bin {
        let distance: Float
        let height: Float
    }

    private var history: [TerrainResult] = []
    private var lastStableKind: TerrainKind = .unknown
    private var lastCameraBands = 0
    /// Consecutive frames where the depth profile was too poor to classify.
    /// Used to keep the previous verdict alive for a short while instead of
    /// flickering to "路况待确认" on every transient dropout.
    private var degradedFrames = 0
    private var obstacleStreak = 0
    private var obstacleHold = 0
    private var obstacleDistance: Float?
    private var obstacleConfirmed = false
    private var obstacleDetail = ""
    /// Stair direction is latched: weak "keep me on stairs" evidence may never
    /// flip up/down on its own, only repeated strict geometry can.
    private var stairDirection: Float = 0
    private var reversalVotes = 0

    private var onStairs: Bool {
        lastStableKind == .stairsUp || lastStableKind == .stairsDown
    }

    func reset() {
        history.removeAll()
        lastStableKind = .unknown
        lastCameraBands = 0
        degradedFrames = 0
        obstacleStreak = 0
        obstacleHold = 0
        obstacleDistance = nil
        obstacleConfirmed = false
        stairDirection = 0
        reversalVotes = 0
    }

    func unavailable(_ message: String) -> TerrainResult {
        stabilize(TerrainResult(kind: .unknown, confidence: 0.1, debug: message, distance: nil))
    }

    func analyze(frame: ARFrame, displayTransform: CGAffineTransform) -> TerrainResult? {
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return nil }
        let depthMap = depthData.depthMap
        guard CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32 else {
            return nil
        }

        let depthWidth = CVPixelBufferGetWidth(depthMap)
        let depthHeight = CVPixelBufferGetHeight(depthMap)
        guard CVPixelBufferLockBaseAddress(depthMap, .readOnly) == kCVReturnSuccess else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let depthBase = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        var lockedConfidenceMap: CVPixelBuffer?
        if let confidenceMap = depthData.confidenceMap,
           CVPixelBufferGetPixelFormatType(confidenceMap) == kCVPixelFormatType_OneComponent8,
           CVPixelBufferGetWidth(confidenceMap) == depthWidth,
           CVPixelBufferGetHeight(confidenceMap) == depthHeight,
           CVPixelBufferLockBaseAddress(confidenceMap, .readOnly) == kCVReturnSuccess {
            lockedConfidenceMap = confidenceMap
        }
        defer {
            if let lockedConfidenceMap = lockedConfidenceMap {
                CVPixelBufferUnlockBaseAddress(lockedConfidenceMap, .readOnly)
            }
        }
        let depthPointer = depthBase.assumingMemoryBound(to: Float.self)
        let depthStride = CVPixelBufferGetBytesPerRow(depthMap) / MemoryLayout<Float>.stride
        let confidencePointer = lockedConfidenceMap.flatMap { CVPixelBufferGetBaseAddress($0) }?
            .assumingMemoryBound(to: UInt8.self)
        let confidenceStride = lockedConfidenceMap.map {
            CVPixelBufferGetBytesPerRow($0) / MemoryLayout<UInt8>.stride
        } ?? 0

        let columns = 7
        let rows = 9
        let imageWidth = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let imageHeight = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let intrinsics = frame.camera.intrinsics
        let cameraTransform = frame.camera.transform
        let cameraPosition = cameraTransform.columns.3
        var samples: [Sample] = []
        var rowDepths = Array(repeating: [Float](), count: rows)
        samples.reserveCapacity(columns * rows)
        let viewToImage = displayTransform.inverted()

        // A single pixel is fragile: LiDAR holes and low-confidence speckles are
        // common indoors. Taking the median of a 3x3 patch lifts the usable
        // sample rate sharply, which is what removes most "样本不足" verdicts.
        func patchDepth(_ u: Float, _ v: Float) -> Float? {
            let centerX = min(depthWidth - 1, max(0, Int(u * Float(depthWidth - 1))))
            let centerY = min(depthHeight - 1, max(0, Int(v * Float(depthHeight - 1))))
            var values: [Float] = []
            values.reserveCapacity(9)
            for dy in -1...1 {
                let y = centerY + dy
                if y < 0 || y >= depthHeight { continue }
                for dx in -1...1 {
                    let x = centerX + dx
                    if x < 0 || x >= depthWidth { continue }
                    let depth = depthPointer[y * depthStride + x]
                    guard depth.isFinite, depth >= 0.25, depth <= 6.0 else { continue }
                    if let confidencePointer = confidencePointer,
                       confidencePointer[y * confidenceStride + x] == 0 { continue }
                    values.append(depth)
                }
            }
            return median(values)
        }

        for row in 0..<rows {
            let viewV = 0.28 + 0.66 * CGFloat(row) / CGFloat(rows - 1)
            for column in 0..<columns {
                let viewU = 0.30 + 0.40 * CGFloat(column) / CGFloat(columns - 1)
                let imagePoint = CGPoint(x: viewU, y: viewV).applying(viewToImage)
                let u = Float(imagePoint.x)
                let v = Float(imagePoint.y)
                guard u >= 0, u <= 1, v >= 0, v <= 1 else { continue }
                guard let depth = patchDepth(u, v) else { continue }
                rowDepths[row].append(depth)

                let pixelX = u * imageWidth
                let pixelY = v * imageHeight
                let rayX = (pixelX - intrinsics[2, 0]) / intrinsics[0, 0]
                let rayY = -(pixelY - intrinsics[2, 1]) / intrinsics[1, 1]
                let cameraPoint = SIMD3<Float>(rayX * depth, rayY * depth, -depth)
                // viewU already constrains the central walking corridor in portrait space.
                // Raw camera X is not the user's horizontal axis when the UI is portrait.
                let worldPoint = cameraTransform * SIMD4<Float>(cameraPoint, 1.0)
                let forward = hypot(worldPoint.x - cameraPosition.x, worldPoint.z - cameraPosition.z)
                let height = worldPoint.y - cameraPosition.y
                guard forward >= 0.4, forward <= 4.0, height >= -3.0, height <= 2.5 else { continue }
                samples.append(Sample(forward: forward, height: height, depth: depth))
            }
        }

        // Camera evidence: count stacked horizontal tread edges in the live image.
        lastCameraBands = cameraStairBands(frame)

        if let dropOff = detectDropOff(rowDepths: rowDepths) {
            degradedFrames = 0
            return stabilize(TerrainResult(
                kind: .dropOff,
                confidence: dropOff,
                debug: "相邻区域出现显著深度突变，可能存在临空边缘或遮挡",
                distance: median(samples.map { $0.depth })
            ))
        }

        // Obstacle check runs on its own narrow centre corridor, before the
        // floor-plane fit: a wall filling the view has a flat-looking profile
        // and would otherwise be reported as "平路".
        if let report = detectObstacle(frame: frame, displayTransform: displayTransform, depthAt: patchDepth) {
            obstacleStreak += 1
            obstacleDistance = report.distance
            obstacleDetail = "高差 \(String(format: "%.2f", report.heightSpan))m · 支撑 \(report.support) 点"
            obstacleHold = 5
            if obstacleStreak >= 2 { obstacleConfirmed = true }
        } else {
            obstacleStreak = 0
            if obstacleHold > 0 {
                obstacleHold -= 1
            } else {
                obstacleDistance = nil
                obstacleConfirmed = false
            }
        }
        if obstacleConfirmed, let obstacle = obstacleDistance {
            degradedFrames = 0
            return TerrainResult(
                kind: .obstacle,
                confidence: min(0.93, 0.60 + (2.8 - min(2.8, obstacle)) * 0.11),
                debug: "竖直面障碍 \(String(format: "%.1f", obstacle))m · \(obstacleDetail)",
                distance: obstacle
            )
        }

        let validCount = samples.count
        guard validCount >= 12 else {
            return held(
                "有效深度样本不足（\(validCount)）",
                confidence: 0.2,
                distance: median(samples.map { $0.depth })
            )
        }

        // Only the near-to-mid field decides the walking surface. Distant bins
        // are usually walls, doors or railings, and including them is what used
        // to flip a descent into "上楼梯".
        var binValues = Array(repeating: [Sample](), count: 8)
        for sample in samples where sample.forward <= 2.6 {
            let index = min(7, max(0, Int((sample.forward - 0.4) / 0.3)))
            binValues[index].append(sample)
        }
        var bins: [Bin] = []
        for values in binValues where values.count >= 3 {
            let heights = values.map { $0.height }
            let distances = values.map { $0.forward }
            guard let binHeight = median(heights), let binDistance = median(distances) else { continue }
            bins.append(Bin(distance: binDistance, height: binHeight))
        }
        guard bins.count >= 2 else {
            return held(
                "前方地面剖面不完整，仅有 \(bins.count) 个有效区段",
                confidence: 0.25,
                distance: median(samples.map { $0.depth })
            )
        }

        guard let fit = fitProfile(bins) else {
            return held("无法拟合地面坡度", confidence: 0.2, distance: median(samples.map { $0.depth }))
        }
        let slope = fit.slope
        let residual = fit.residual
        let profileBins = fit.bins
        let deltas = zip(profileBins.dropFirst(), profileBins).map { pair in
            pair.0.height - pair.1.height
        }
        let direction: Float = slope >= 0 ? 1.0 : -1.0
        let informative = deltas.filter { abs($0) >= 0.02 }
        let monotonic = informative.isEmpty ? 0 : Float(informative.filter { $0 * direction > 0 }.count) / Float(informative.count)
        let jumps = deltas.filter { abs($0) >= 0.09 }.count
        let degrees = atan(slope) * 180.0 / .pi
        let coverage = Float(validCount) / Float(columns * rows)
        let evidence = min(0.88, 0.40 + 0.25 * coverage + 0.20 * min(1.0, Float(bins.count) / 6.0))
        // A two-bin profile always fits a line perfectly, so its residual is not
        // trustworthy as evidence; cap the confidence in that case.
        let thinProfile = profileBins.count < 3
        let distance = median(samples.map { $0.depth })
        let debug = "坡度 \(String(format: "%.1f", degrees))° · 跳变 \(jumps) · 残差 \(String(format: "%.2f", residual))m · 样本 \(validCount) · 区段 \(profileBins.count)/\(bins.count)"

        let raw: TerrainResult?
        if abs(degrees) <= 5.0 && residual <= 0.14 {
            raw = TerrainResult(
                kind: .flat,
                confidence: thinProfile ? min(evidence, 0.55) : evidence,
                debug: debug,
                distance: distance
            )
        } else {
            // Full geometric stair evidence (strict, as before).
            let stairEvidence = residual >= 0.05 && jumps >= 2 && abs(degrees) >= 8 && abs(degrees) <= 45 && monotonic >= 0.60
            // Continuation: once on stairs, a much weaker profile keeps the state
            // so mid-stair frames stop reporting "unknown".
            let stairContinue = onStairs && residual >= 0.04 && jumps >= 1
                && abs(degrees) >= 5 && abs(degrees) <= 45 && monotonic >= 0.45
            // Camera fusion: stacked tread edges in the live image can confirm
            // stairs when the depth profile alone is borderline.
            let stairCamera = lastCameraBands >= 2 && residual >= 0.04 && jumps >= 1
                && abs(degrees) >= 5 && abs(degrees) <= 45 && monotonic >= 0.45
            // Loosest tier: a single visible tread band plus any depth break.
            let stairLoose = lastCameraBands >= 1 && residual >= 0.04 && jumps >= 1
                && abs(degrees) >= 5 && abs(degrees) <= 45 && monotonic >= 0.50
            let slopeEvidence = residual <= 0.13 && abs(degrees) >= 4.5 && abs(degrees) <= 35 && monotonic >= 0.50
            if stairEvidence || stairContinue || stairCamera || stairLoose {
                let measuredSign: Float = slope > 0 ? 1 : -1
                if stairEvidence {
                    if stairDirection == 0 {
                        stairDirection = measuredSign
                        reversalVotes = 0
                    } else if measuredSign != stairDirection {
                        // Six consecutive strict frames (~0.6s) are needed to
                        // reverse a latched direction; noise cannot do it.
                        // A clearly measured steep opposite slope counts double.
                        reversalVotes += abs(degrees) >= 14 ? 2 : 1
                        if reversalVotes >= 6 {
                            stairDirection = measuredSign
                            reversalVotes = 0
                        }
                    } else {
                        reversalVotes = 0
                    }
                }
                // Continuation / camera-only evidence keeps the current
                // direction instead of being allowed to flip it.
                let announcedSign = stairDirection == 0 ? measuredSign : stairDirection
                let cameraBonus = 0.02 * Float(min(4, lastCameraBands))
                let continueBonus: Float = (stairContinue || stairLoose) && !stairEvidence ? 0.02 : 0
                raw = TerrainResult(
                    kind: announcedSign > 0 ? .stairsUp : .stairsDown,
                    confidence: min(0.94, evidence + 0.08 + cameraBonus + continueBonus),
                    debug: debug + " · 视觉台阶边 \(lastCameraBands) 组",
                    distance: distance
                )
            } else if slopeEvidence {
                raw = TerrainResult(
                    kind: slope > 0 ? .uphill : .downhill,
                    confidence: thinProfile ? min(evidence, 0.55) : evidence,
                    debug: debug,
                    distance: distance
                )
            } else {
                raw = nil
            }
        }

        let isStairRaw = raw?.kind == .stairsUp || raw?.kind == .stairsDown
        if !isStairRaw && !onStairs {
            stairDirection = 0
            reversalVotes = 0
        }
        guard let raw = raw else {
            return held("剖面混杂，暂无可判定的路面结构", confidence: 0.30, distance: distance)
        }
        degradedFrames = 0
        return stabilize(raw)
    }

    /// Keeps the previous verdict for a short grace period instead of
    /// announcing "路况待确认" on every isolated bad frame.
    private func held(_ reason: String, confidence: Float, distance: Float?) -> TerrainResult {
        degradedFrames += 1
        let previous = lastStableKind
        let canHold = previous != .unknown && previous != .dropOff && previous != .obstacle
        if canHold && degradedFrames <= 12 {
            return stabilize(TerrainResult(
                kind: previous,
                confidence: max(0.45, confidence),
                debug: reason + " · 延续上一判定",
                distance: distance
            ))
        }
        return stabilize(TerrainResult(kind: .unknown, confidence: max(0.15, confidence), debug: reason, distance: distance))
    }

    private struct ProfileFit {
        let slope: Float
        let intercept: Float
        let residual: Float
        let bins: [Bin]
    }

    /// Theil-Sen fit followed by one trimming pass. Walls, doors and railings
    /// behind / beyond the steps produce residuals far larger than a stair
    /// tread, so they are removed before the slope sign is trusted.
    private func fitProfile(_ input: [Bin]) -> ProfileFit? {
        var current = input
        var result: ProfileFit?
        for _ in 0..<2 {
            guard current.count >= 2 else { return result }
            var pairSlopes: [Float] = []
            for first in 0..<current.count {
                for second in (first + 1)..<current.count {
                    let deltaDistance = current[second].distance - current[first].distance
                    if abs(deltaDistance) > 0.08 {
                        pairSlopes.append((current[second].height - current[first].height) / deltaDistance)
                    }
                }
            }
            guard let slope = median(pairSlopes) else { return result }
            let intercept = median(current.map { $0.height - slope * $0.distance }) ?? 0
            let errors = current.map { abs($0.height - (slope * $0.distance + intercept)) }
            let residual = sqrt(errors.reduce(Float(0)) { $0 + $1 * $1 } / Float(errors.count))
            result = ProfileFit(slope: slope, intercept: intercept, residual: residual, bins: current)
            let limit = max(0.28, residual * 2.2)
            let inliers = zip(current, errors).filter { $0.1 <= limit }.map { $0.0 }
            if inliers.count == current.count || inliers.count < 3 { break }
            current = inliers
        }
        return result
    }

    private struct ObstacleReport {
        let distance: Float
        let heightSpan: Float
        let support: Int
    }

    /// Obstacle = a vertical surface inside the walking corridor.
    ///
    /// Works in world coordinates, not in raw depth: on a wall, many points
    /// share almost the same forward distance but span a large height range.
    /// On the floor it is the opposite (forward spread is large, height is
    /// nearly constant), so this test does not depend on how the phone is
    /// tilted and does not fire on steep floor views.
    private func detectObstacle(
        frame: ARFrame,
        displayTransform: CGAffineTransform,
        depthAt: (Float, Float) -> Float?
    ) -> ObstacleReport? {
        let imageWidth = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let imageHeight = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let intrinsics = frame.camera.intrinsics
        let cameraTransform = frame.camera.transform
        let cameraPosition = cameraTransform.columns.3
        let viewToImage = displayTransform.inverted()

        let rowCount = 15
        let columnCount = 7
        let binCount = 18
        let binWidth: Float = 0.15
        let firstDistance: Float = 0.45
        var binForward = Array(repeating: [Float](), count: binCount)
        var binHeight = Array(repeating: [Float](), count: binCount)
        var binColumns = [Int](repeating: 0, count: binCount)

        for row in 0..<rowCount {
            let viewV = 0.24 + 0.72 * CGFloat(row) / CGFloat(rowCount - 1)
            for column in 0..<columnCount {
                let viewU = 0.36 + 0.28 * CGFloat(column) / CGFloat(columnCount - 1)
                let imagePoint = CGPoint(x: viewU, y: viewV).applying(viewToImage)
                let u = Float(imagePoint.x)
                let v = Float(imagePoint.y)
                guard u >= 0, u <= 1, v >= 0, v <= 1 else { continue }
                guard let depth = depthAt(u, v) else { continue }

                let rayX = (u * imageWidth - intrinsics[2, 0]) / intrinsics[0, 0]
                let rayY = -(v * imageHeight - intrinsics[2, 1]) / intrinsics[1, 1]
                let cameraPoint = SIMD3<Float>(rayX * depth, rayY * depth, -depth)
                let worldPoint = cameraTransform * SIMD4<Float>(cameraPoint, 1.0)
                let forward = hypot(worldPoint.x - cameraPosition.x, worldPoint.z - cameraPosition.z)
                let height = worldPoint.y - cameraPosition.y
                guard forward >= firstDistance, forward <= firstDistance + Float(binCount) * binWidth else { continue }
                let index = min(binCount - 1, Int((forward - firstDistance) / binWidth))
                binForward[index].append(forward)
                binHeight[index].append(height)
                binColumns[index] |= 1 << column
            }
        }

        var best: ObstacleReport?
        for index in 0..<binCount {
            let heights = binHeight[index]
            // A wall is seen by many pixels across several columns; a speckle
            // or a single railing bar is not enough.
            guard heights.count >= 10, binColumns[index].nonzeroBitCount >= 3 else { continue }
            let sorted = heights.sorted()
            let lowIndex = min(sorted.count - 1, Int(Float(sorted.count) * 0.10))
            let highIndex = min(sorted.count - 1, Int(Float(sorted.count) * 0.90))
            let span = sorted[highIndex] - sorted[lowIndex]
            guard span >= 0.45 else { continue }
            guard let distance = median(binForward[index]) else { continue }

            if onStairs {
                // On stairs only very close, genuinely tall obstructions are
                // worth interrupting for; risers and the far stairwell wall
                // must not turn into constant obstacle alarms.
                guard distance >= 0.9, distance <= 1.6, span >= 0.60 else { continue }
            } else if lastCameraBands >= 2 && span < 0.80 {
                // Many visible tread edges plus a modest height spread reads
                // as a staircase, not as an obstacle.
                continue
            }

            let report = ObstacleReport(distance: distance, heightSpan: span, support: heights.count)
            if best == nil || distance < best!.distance { best = report }
        }
        return best
    }

    /// Counts distinct stacked horizontal edge bands (stair treads) in the
    /// camera image. Captured buffers are landscape; a horizontal line in the
    /// portrait view is vertical in the sensor, so world-horizontal treads
    /// produce a strong gradient along the sensor Y axis.
    private func cameraStairBands(_ frame: ARFrame) -> Int {
        let pixelBuffer = frame.capturedImage
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            || format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange else { return 0 }
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess else { return 0 }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return 0 }

        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let luma = base.assumingMemoryBound(to: UInt8.self)

        let step = 8
        let x0 = width * 2 / 10, x1 = width * 8 / 10
        let y0 = height * 15 / 100, y1 = height * 85 / 100
        guard x1 - x0 > 3 * step, y1 - y0 > 3 * step else { return 0 }

        var bandCounts = [Int](repeating: 0, count: height / step + 2)
        var y = y0 + step
        while y < y1 - step {
            let rowBase = y * stride
            var x = x0 + step
            while x < x1 - step {
                let center = Int(luma[rowBase + x])
                let gy = Int(luma[rowBase - stride + x]) - Int(luma[rowBase + stride + x])
                let gx = Int(luma[rowBase + x - step]) - Int(luma[rowBase + x + step])
                if abs(gy) > 26 && abs(gy) > 2 * abs(gx) {
                    bandCounts[y / step] += 1
                }
                _ = center
                x += step
            }
            y += step
        }

        // Cluster band rows separated by gaps into distinct tread groups.
        var bands = 0
        var inBand = false
        var gap = 0
        for count in bandCounts {
            if count >= 4 {
                if !inBand { bands += 1; inBand = true }
                gap = 0
            } else if inBand {
                gap += 1
                if gap >= 2 { inBand = false }
            }
        }
        return bands
    }

    private func detectDropOff(rowDepths: [[Float]]) -> Float? {
        var strongest: Float?
        for index in 0..<(rowDepths.count - 1) {
            // Device pitch changes which side of an edge is nearer. Treat either
            // direction of a large, abrupt depth transition as a stop-and-check hazard.
            let upperValues = rowDepths[index]
            let lowerValues = rowDepths[index + 1]
            guard upperValues.count >= 5, lowerValues.count >= 5,
                  let upper = median(upperValues), let lower = median(lowerValues) else { continue }
            let near = min(upper, lower)
            let far = max(upper, lower)
            let viewV = 0.28 + 0.66 * Float(index + 1) / Float(rowDepths.count - 1)
            guard viewV >= 0.50, near >= 0.35, near <= 2.5, far >= 3.3,
                  far - near >= 1.0, far / near >= 1.65 else { continue }
            let score = min(0.90, 0.58 + Float(far - near) * 0.08)
            strongest = max(strongest ?? 0, score)
        }
        return strongest
    }

    private func stabilize(_ raw: TerrainResult) -> TerrainResult {
        history.append(raw)
        if history.count > 8 { history.removeFirst() }
        let clearWindow = history.suffix(9)
        if clearWindow.count == 9 && clearWindow.allSatisfy({ $0.kind == .unknown }) {
            lastStableKind = .unknown
            return TerrainResult(
                kind: .unknown,
                confidence: raw.confidence,
                debug: raw.debug + " · 已清除旧路况",
                distance: raw.distance
            )
        }
        let window = history.suffix(6)
        var weights: [TerrainKind: Float] = [:]
        var counts: [TerrainKind: Int] = [:]
        for result in window {
            weights[result.kind, default: 0] += max(0.1, result.confidence)
            counts[result.kind, default: 0] += 1
        }
        guard let candidate = weights.max(by: { $0.value < $1.value })?.key,
              let count = counts[candidate] else { return raw }
        let totalWeight = weights.values.reduce(0, +)
        let share = (weights[candidate] ?? 0) / max(totalWeight, 0.001)
        let reversingStairs = (lastStableKind == .stairsUp && candidate == .stairsDown) ||
            (lastStableKind == .stairsDown && candidate == .stairsUp)
        let requiredCount = reversingStairs ? 5 : (candidate == .dropOff ? 2 : 3)
        let candidateResults = window.filter { $0.kind == candidate }
        let meanConfidence = candidateResults.reduce(Float(0)) { $0 + $1.confidence } / Float(max(1, candidateResults.count))

        guard count >= requiredCount, share >= 0.50, meanConfidence >= 0.40 else {
            // While waiting for confirmation, keep showing the last stable
            // verdict rather than the placeholder.
            let shown: TerrainKind = lastStableKind == .unknown ? .unknown : lastStableKind
            return TerrainResult(kind: shown, confidence: meanConfidence, debug: raw.debug + " · 正在确认", distance: raw.distance)
        }
        lastStableKind = candidate
        return TerrainResult(
            kind: candidate,
            confidence: meanConfidence,
            debug: raw.debug + " · 稳定 \(count) 帧",
            distance: raw.distance
        )
    }
}

private func median(_ values: [Float]) -> Float? {
    guard !values.isEmpty else { return nil }
    let sorted = values.sorted()
    let middle = sorted.count / 2
    if sorted.count % 2 == 0 {
        return (sorted[middle - 1] + sorted[middle]) / 2
    }
    return sorted[middle]
}
