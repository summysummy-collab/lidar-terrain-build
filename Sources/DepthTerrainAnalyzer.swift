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
        if let obstacle = detectObstacle(frame: frame, displayTransform: displayTransform, depthAt: patchDepth) {
            obstacleStreak += 1
            obstacleDistance = obstacle
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
                debug: "中央通道出现竖直面 · 障碍 \(String(format: "%.1f", obstacle))m",
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

        var binValues = Array(repeating: [Sample](), count: 9)
        for sample in samples {
            let index = min(8, max(0, Int((sample.forward - 0.4) / 0.4)))
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

        var pairSlopes: [Float] = []
        for first in 0..<bins.count {
            for second in (first + 1)..<bins.count {
                let deltaDistance = bins[second].distance - bins[first].distance
                if abs(deltaDistance) > 0.08 {
                    pairSlopes.append((bins[second].height - bins[first].height) / deltaDistance)
                }
            }
        }
        guard let slope = median(pairSlopes) else {
            return held("无法拟合地面坡度", confidence: 0.2, distance: median(samples.map { $0.depth }))
        }
        let intercept = median(bins.map { $0.height - slope * $0.distance }) ?? 0
        let residual = sqrt(bins.reduce(Float(0)) { partial, bin in
            let error = bin.height - (slope * bin.distance + intercept)
            return partial + error * error
        } / Float(bins.count))
        let deltas = zip(bins.dropFirst(), bins).map { pair in
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
        let thinProfile = bins.count < 3
        let distance = median(samples.map { $0.depth })
        let debug = "坡度 \(String(format: "%.1f", degrees))° · 跳变 \(jumps) · 残差 \(String(format: "%.2f", residual))m · 样本 \(validCount)"

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
            let stairLoose = lastCameraBands >= 1 && residual >= 0.03 && jumps >= 1
                && abs(degrees) >= 5 && abs(degrees) <= 45 && monotonic >= 0.40
            let slopeEvidence = residual <= 0.13 && abs(degrees) >= 4.5 && abs(degrees) <= 35 && monotonic >= 0.50
            if stairEvidence || stairContinue || stairCamera || stairLoose {
                let cameraBonus = 0.02 * Float(min(4, lastCameraBands))
                let continueBonus: Float = (stairContinue || stairLoose) && !stairEvidence ? 0.02 : 0
                raw = TerrainResult(
                    kind: slope > 0 ? .stairsUp : .stairsDown,
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

    /// Detects a vertical surface (wall, cabinet, parked bike, box, pillar)
    /// standing inside the walking corridor.
    ///
    /// A vertical surface returns almost the same depth for many consecutive
    /// image rows while its height above the camera changes a lot. Floor pixels
    /// are the opposite: depth changes rapidly from row to row while height
    /// stays on one plane. Both conditions are required, which keeps steep
    /// floor views from being mistaken for obstacles.
    private func detectObstacle(
        frame: ARFrame,
        displayTransform: CGAffineTransform,
        depthAt: (Float, Float) -> Float?
    ) -> Float? {
        let imageWidth = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let imageHeight = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let intrinsics = frame.camera.intrinsics
        let cameraTransform = frame.camera.transform
        let cameraPosition = cameraTransform.columns.3
        let viewToImage = displayTransform.inverted()

        let rowCount = 13
        let columnCount = 5
        var rowDepths = Array(repeating: [Float](), count: rowCount)
        var rowHeights = Array(repeating: [Float](), count: rowCount)

        for row in 0..<rowCount {
            let viewV = 0.26 + 0.70 * CGFloat(row) / CGFloat(rowCount - 1)
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
                rowDepths[row].append(depth)
                rowHeights[row].append(worldPoint.y - cameraPosition.y)
            }
        }

        var depthMedians = [Float?](repeating: nil, count: rowCount)
        var heightMedians = [Float?](repeating: nil, count: rowCount)
        var breadth = [Float](repeating: 0, count: rowCount)
        for row in 0..<rowCount {
            let values = rowDepths[row]
            guard values.count >= 3, let value = median(values) else { continue }
            depthMedians[row] = value
            heightMedians[row] = median(rowHeights[row])
            breadth[row] = Float(values.filter { abs($0 - value) <= value * 0.22 }.count) / Float(values.count)
        }

        let minimumRows = onStairs ? 4 : 3
        var nearest: Float?
        var index = 0
        while index < rowCount {
            guard let base = depthMedians[index], breadth[index] >= 0.60 else {
                index += 1
                continue
            }
            var end = index
            var runningMin = base
            var runningMax = base
            while end + 1 < rowCount,
                  let next = depthMedians[end + 1], breadth[end + 1] >= 0.60 {
                let newMin = min(runningMin, next)
                let newMax = max(runningMax, next)
                if newMax / max(newMin, 0.01) > 1.18 { break }
                end += 1
                runningMin = newMin
                runningMax = newMax
            }

            let runLength = end - index + 1
            let runDepths = depthMedians[index...end].compactMap { $0 }
            let plateau = median(runDepths) ?? base
            let runHeights = heightMedians[index...end].compactMap { $0 }
            let heightSpan = (runHeights.max() ?? 0) - (runHeights.min() ?? 0)
            let startsAtTop = index == 0
            let endsAtBottom = end == rowCount - 1

            if runLength >= minimumRows,
               plateau >= 0.55, plateau <= 3.0,
               heightSpan >= 0.25 {
                var accepted = false
                if startsAtTop {
                    // Wall filling the upper view: needs a floor strip nearer
                    // than the wall (otherwise it is a steep floor view), and
                    // no tread bands that would explain it as a staircase.
                    if endsAtBottom {
                        accepted = lastCameraBands <= 1 && plateau <= 1.6 && !onStairs
                    } else if let below = depthMedians[end + 1], below <= plateau * 0.92 {
                        accepted = lastCameraBands <= 1 && !(onStairs && plateau < 1.1)
                    }
                } else if let beyond = depthMedians[index - 1], beyond >= plateau * 1.30 {
                    // Object on the floor: the floor / space behind it is
                    // clearly farther than the object face.
                    accepted = !(onStairs && plateau < 1.0)
                }
                if accepted {
                    nearest = min(nearest ?? .greatestFiniteMagnitude, plateau)
                }
            }
            index = max(end + 1, index + 1)
        }
        return nearest
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
