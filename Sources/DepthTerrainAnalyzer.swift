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

    func reset() {
        history.removeAll()
        lastStableKind = .unknown
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

        let columns = 9
        let rows = 13
        let imageWidth = Float(CVPixelBufferGetWidth(frame.capturedImage))
        let imageHeight = Float(CVPixelBufferGetHeight(frame.capturedImage))
        let intrinsics = frame.camera.intrinsics
        let cameraTransform = frame.camera.transform
        let cameraPosition = cameraTransform.columns.3
        var samples: [Sample] = []
        var rowDepths = Array(repeating: [Float](), count: rows)
        samples.reserveCapacity(columns * rows)
        let viewToImage = displayTransform.inverted()

        for row in 0..<rows {
            let viewV = 0.28 + 0.66 * CGFloat(row) / CGFloat(rows - 1)
            for column in 0..<columns {
                let viewU = 0.30 + 0.40 * CGFloat(column) / CGFloat(columns - 1)
                let imagePoint = CGPoint(x: viewU, y: viewV).applying(viewToImage)
                let u = Float(imagePoint.x)
                let v = Float(imagePoint.y)
                guard u >= 0, u <= 1, v >= 0, v <= 1 else { continue }
                let x = min(depthWidth - 1, max(0, Int(u * Float(depthWidth - 1))))
                let y = min(depthHeight - 1, max(0, Int(v * Float(depthHeight - 1))))
                let depth = depthPointer[y * depthStride + x]
                guard depth.isFinite, depth >= 0.25, depth <= 6.0 else { continue }

                if let confidencePointer = confidencePointer {
                    let confidence = confidencePointer[y * confidenceStride + x]
                    if confidence == 0 { continue }
                }
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

        if let dropOff = detectDropOff(rowDepths: rowDepths) {
            return stabilize(TerrainResult(
                kind: .dropOff,
                confidence: dropOff,
                debug: "相邻区域出现显著深度突变，可能存在临空边缘或遮挡",
                distance: median(samples.map { $0.depth })
            ))
        }

        let validCount = samples.count
        guard validCount >= 24 else {
            return stabilize(TerrainResult(
                kind: .unknown,
                confidence: 0.2,
                debug: "有效深度样本不足（\(validCount)），请对准前方地面",
                distance: median(samples.map { $0.depth })
            ))
        }

        var binValues = Array(repeating: [Sample](), count: 9)
        for sample in samples {
            let index = min(8, max(0, Int((sample.forward - 0.4) / 0.4)))
            binValues[index].append(sample)
        }
        var bins: [Bin] = []
        for (index, values) in binValues.enumerated() where values.count >= 3 {
            let heights = values.map { $0.height }
            let distances = values.map { $0.forward }
            guard let binHeight = median(heights), let binDistance = median(distances) else { continue }
            bins.append(Bin(distance: binDistance, height: binHeight))
            _ = index
        }
        guard bins.count >= 4 else {
            return stabilize(TerrainResult(
                kind: .unknown,
                confidence: 0.25,
                debug: "前方地面剖面不完整，仅有 \(bins.count) 个有效区段",
                distance: median(samples.map { $0.depth })
            ))
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
        guard let slope = median(pairSlopes) else { return nil }
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
        let distance = median(samples.map { $0.depth })
        let debug = "坡度 \(String(format: "%.1f", degrees))° · 跳变 \(jumps) · 残差 \(String(format: "%.2f", residual))m · 样本 \(validCount)"

        let raw: TerrainResult
        if abs(degrees) <= 4.5 && residual <= 0.12 {
            raw = TerrainResult(kind: .flat, confidence: evidence, debug: debug, distance: distance)
        } else {
            let stairEvidence = residual >= 0.05 && jumps >= 2 && abs(degrees) >= 8 && abs(degrees) <= 45 && monotonic >= 0.60
            let slopeEvidence = residual <= 0.11 && abs(degrees) >= 5 && abs(degrees) <= 35 && monotonic >= 0.55
            if stairEvidence {
                raw = TerrainResult(
                    kind: slope > 0 ? .stairsUp : .stairsDown,
                    confidence: min(0.92, evidence + 0.08),
                    debug: debug,
                    distance: distance
                )
            } else if slopeEvidence {
                raw = TerrainResult(kind: slope > 0 ? .uphill : .downhill, confidence: evidence, debug: debug, distance: distance)
            } else {
                raw = TerrainResult(kind: .unknown, confidence: 0.35, debug: debug, distance: distance)
            }
        }
        return stabilize(raw)
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
        let recent = history.suffix(3)
        if recent.count == 3 && recent.allSatisfy({ $0.kind == .unknown }) {
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

        guard count >= requiredCount, share >= 0.50, meanConfidence >= 0.45 else {
            return TerrainResult(kind: lastStableKind, confidence: meanConfidence, debug: raw.debug + " · 正在确认", distance: raw.distance)
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
