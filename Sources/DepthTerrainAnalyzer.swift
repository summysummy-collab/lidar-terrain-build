import ARKit
import CoreGraphics
import Foundation
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
    /// Seconds this verdict has been continuously held by the stability gate.
    /// A downstream actuator uses it to decide how hard it can commit.
    var heldSeconds: Double = 0
    /// The state the gate is currently converging on, if any. Lets a moving
    /// platform start slowing down before the verdict actually flips.
    var pendingKind: TerrainKind? = nil
    /// 0...1 progress towards `pendingKind`.
    var pendingProgress: Float = 0
    /// Measured height of a single step, when a staircase was recognised.
    var stepHeight: Float? = nil
    /// How many treads the probe could see at once.
    var treadCount: Int = 0
    /// Set when the obstacle is a riser too tall to climb, so the announcement
    /// can say "台阶过高" instead of the generic wall warning.
    var tallStep: Float? = nil

    var announcement: String {
        switch kind {
        case .unknown: return "前方路况待确认，请减速"
        case .flat: return "前方平路"
        case .uphill: return "前方上坡"
        case .downhill: return "前方下坡，注意脚下"
        case .stairsUp, .stairsDown:
            let direction = kind == .stairsUp ? "上" : "下"
            if let step = stepHeight {
                return "前方\(direction)楼梯，台阶高约 \(Int((step * 100).rounded())) 厘米"
            }
            return "前方\(direction)楼梯，注意台阶"
        case .obstacle:
            if let riser = tallStep {
                let centimetres = Int((riser * 100).rounded())
                if let distance = distance {
                    return "注意，正前方 \(String(format: "%.1f", distance)) 米有 \(centimetres) 厘米高台阶，无法通行，请停下"
                }
                return "注意，前方有 \(centimetres) 厘米高台阶，无法通行，请停下"
            }
            guard let distance = distance else { return "注意，前方有障碍" }
            let rounded = min(0.5, max(0.15, distance))
            return "注意，正前方 \(String(format: "%.1f", rounded)) 米有障碍，请停下"
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
    /// Last measured riser height, kept so a one-frame dropout mid-flight does
    /// not blank the step size on screen.
    private var lastStepHeight: Float?
    private var lastTreadCount = 0
    /// Consecutive frames where the probe measured a riser taller than the
    /// climbable limit. Two frames keep a single mis-measured pair of treads
    /// from stopping the platform on a normal flight.
    private var tallStepStreak = 0

    // MARK: - Fused motion (accelerometer + gyroscope + barometer)

    private struct MotionSample {
        let time: TimeInterval
        let horizontal: SIMD2<Float>
        let altitude: Double
    }

    private var motionTrail: [MotionSample] = []
    /// Barometer slope in tan form (rise / run), smoothed.
    private var motionSlope: Float = 0
    private var motionSlopeValid = false
    private var motionSlopeUpdatedAt: TimeInterval = -1
    private var motionSlopeStreak = 0
    /// While non-zero, the phone's own vertical travel overrides the sign the
    /// depth profile suggests. Cleared when the two agree again.
    private var motionOverrideSign: Float = 0
    private var motionOverrideUntil: TimeInterval = 0
    private var motionConflictFrames = 0
    private var motionNote = "运动 待采集"

    private var onStairs: Bool {
        lastStableKind == .stairsUp || lastStableKind == .stairsDown
    }

    private let gate = TerrainStabilityGate()

    func reset() {
        gate.reset()
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
        lastStepHeight = nil
        lastTreadCount = 0
        tallStepStreak = 0
        motionTrail.removeAll()
        motionSlope = 0
        motionSlopeValid = false
        motionSlopeUpdatedAt = -1
        motionSlopeStreak = 0
        motionOverrideSign = 0
        motionOverrideUntil = 0
        motionConflictFrames = 0
        motionNote = "运动 待采集"
    }

    func unavailable(_ message: String) -> TerrainResult {
        gate.process(
            stabilize(TerrainResult(kind: .unknown, confidence: 0.1, debug: message, distance: nil)),
            at: ProcessInfo.processInfo.systemUptime
        )
    }

    /// Public entry point. Every verdict — including "no data at all" — passes
    /// through the stability gate, so the caller only ever sees commands with a
    /// dwell time attached rather than raw per-frame opinions.
    func analyze(
        frame: ARFrame,
        displayTransform: CGAffineTransform,
        motion: MotionSnapshot = .unavailable
    ) -> TerrainResult? {
        guard let raw = analyzeRaw(frame: frame, displayTransform: displayTransform, motion: motion) else {
            return nil
        }
        return gate.process(raw, at: frame.timestamp)
    }

    private func analyzeRaw(
        frame: ARFrame,
        displayTransform: CGAffineTransform,
        motion: MotionSnapshot
    ) -> TerrainResult? {
        updateMotion(frame: frame, motion: motion)
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

        // Denser than v6 on purpose: the step-height probe needs enough points
        // per tread to see discrete levels in a height histogram, and more
        // samples per bin also removes the "样本不足" verdicts.
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
            obstacleHold = 4
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
        if obstacleConfirmed, let obstacle = obstacleDistance, obstacle <= 0.50 {
            degradedFrames = 0
            return TerrainResult(
                kind: .obstacle,
                confidence: min(0.95, 0.62 + (0.50 - min(0.50, obstacle)) * 0.36),
                debug: "0.5m 内竖直面障碍 \(String(format: "%.1f", obstacle))m · \(obstacleDetail)",
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

        // Two questions that used to be tangled together are now answered in
        // order:
        //   * WHAT is ahead  -> staircase geometry (discrete levels) vs ramp
        //                       (continuous heights)
        //   * WHICH WAY      -> the sign of the slope, later cross-checked
        //                       against the barometer
        // That is why a ramp with a few paving joints no longer reads as a
        // staircase, and a real flight seen at a shallow angle no longer reads
        // as a ramp.
        let stairProfile = StairGeometryProbe.measure(
            points: samples.map { TerrainPoint(forward: $0.forward, height: $0.height) }
        )
        let treads = stairProfile.treadCount
        let angle = abs(degrees)
        // Levels spread continuously through the height range and nothing
        // discrete to stand on: a ramp wearing a staircase costume. Veto the
        // stair branches instead of trusting how jagged the fitted line is.
        //
        // Only applies to shallow angles. Real ramps are built at 5–12° (an
        // accessible ramp is 1:12, the steepest car-park ramp is about 8.5°); a
        // 30° incline is essentially always a staircase, so a probe that failed
        // to resolve its treads must not demote it to a ramp.
        let rampLike = angle < 22
            && !stairProfile.isLikelyStairs
            && stairProfile.discreteness < 0.42
            && treads <= 2
        let steepLikeStairs = angle >= 22 && angle <= 50
        // A smooth monotone incline with no horizontal level anywhere is a ramp
        // even while the previous verdict was a staircase. Without this escape
        // hatch `onStairs` would latch onto a long slope entered from a flight.
        let strongSlopeEscape = treads <= 1 && residual <= 0.10
            && monotonic >= 0.60 && angle >= 5

        let strongStair = stairProfile.isLikelyStairs
            || (jumps >= 2 && residual >= 0.05 && monotonic >= 0.55
                && angle >= 7 && angle <= 50 && !rampLike)
            || (steepLikeStairs && (jumps >= 1 || lastCameraBands >= 1 || treads >= 2))
        let weakStair = !rampLike && angle >= 4 && angle <= 50
            && ((lastCameraBands >= 2 && jumps >= 1 && residual >= 0.04 && monotonic >= 0.45)
                || (onStairs && !strongSlopeEscape))
        let slopeEvidence = angle >= 3.5 && angle <= 30
            && residual <= 0.13 && monotonic >= 0.48

        let debug = "坡度 \(String(format: "%.1f", degrees))° · 跳变 \(jumps) · 残差 \(String(format: "%.2f", residual))m · 样本 \(validCount) · \(stairProfile.debug)"

        // A landing flattens the profile almost completely. Dropping the latch
        // here means the first frame that sees the new direction is adopted at
        // once, instead of having to out-vote a heading measured before the
        // turn. Without this the old direction survives the entire landing and
        // the reversal only starts counting at the second flight.
        if angle < 3.0 && residual <= 0.10 {
            stairDirection = 0
            reversalVotes = 0
        }

        // A riser taller than the climbable limit is not a staircase, it is a
        // step-up barrier: the platform cannot mount it. This has to be judged
        // before the flat branch, because standing level in front of a raised
        // kerb, a stair-top landing or a high platform yields an almost flat
        // fitted slope — it used to read as "平路" right up to the collision.
        let tallRiser = stairProfile.isTallStep ? stairProfile.stepHeight : nil
        tallStepStreak = tallRiser == nil ? 0 : tallStepStreak + 1
        let frontier = riserFrontier(profileBins)
        let tallStepBarrier: TerrainResult? = {
            guard let riser = tallRiser else { return nil }
            // On a flight the probe occasionally merges two normal treads into
            // one tall gap, so require a longer run there than on open ground.
            let requiredStreak = onStairs ? 4 : 2
            guard tallStepStreak >= requiredStreak else { return nil }
            // The surface must really break, and the break must be near enough
            // to act on. A raised platform 2.5 m away is not a collision yet.
            guard let frontier = frontier, frontier.jump >= 0.15, frontier.distance <= 2.2 else { return nil }
            return TerrainResult(
                kind: .obstacle,
                confidence: min(0.93, 0.54 + 0.40 * stairProfile.confidence),
                debug: "台阶超高 \(Int((riser * 100).rounded()))cm · 前沿 \(String(format: "%.1f", frontier.distance))m · \(stairProfile.debug)",
                distance: frontier.distance,
                treadCount: treads,
                tallStep: riser
            )
        }()

        let raw: TerrainResult?
        if let barrier = tallStepBarrier {
            raw = barrier
        } else if angle <= 4.5 && residual <= 0.12 && !stairProfile.isLikelyStairs {
            raw = TerrainResult(
                kind: .flat,
                confidence: thinProfile ? min(evidence, 0.55) : evidence,
                debug: debug,
                distance: distance
            )
        } else if strongStair || weakStair {
            let measuredSign: Float = slope > 0 ? 1 : -1
            if strongStair {
                if stairDirection == 0 {
                    stairDirection = measuredSign
                    reversalVotes = 0
                } else if measuredSign != stairDirection {
                    // Only strong evidence may reverse. Three votes, with a
                    // steeply measured opposite slope counting double, so a
                    // turn on a landing is adopted in two frames (~0.2 s)
                    // instead of six. The old six-vote bar assumed noise was
                    // the main risk, but by the time strong evidence repeats
                    // three times the surface really has changed.
                    reversalVotes += angle >= 14 ? 2 : 1
                    if reversalVotes >= 3 {
                        stairDirection = measuredSign
                        reversalVotes = 0
                    }
                } else {
                    reversalVotes = 0
                }
            }
            // Continuation / camera-only evidence may hold the current heading
            // but never flip it.
            let announcedSign = stairDirection == 0 ? measuredSign : stairDirection
            let cameraBonus = 0.02 * Float(min(4, lastCameraBands))
            let stepBonus = stairProfile.isLikelyStairs ? 0.10 * stairProfile.confidence : 0
            let weakBonus: Float = strongStair ? 0 : 0.02
            raw = TerrainResult(
                kind: announcedSign > 0 ? .stairsUp : .stairsDown,
                confidence: min(0.95, evidence + 0.10 + stepBonus + cameraBonus + weakBonus),
                debug: debug + " · 视觉台阶边 \(lastCameraBands) 组",
                distance: distance,
                stepHeight: stairProfile.stepHeight,
                treadCount: treads
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

        let fused = fuseMotion(raw: raw, timestamp: frame.timestamp, distance: distance)

        let isStairRaw = fused?.kind == .stairsUp || fused?.kind == .stairsDown
        if !isStairRaw && !onStairs {
            stairDirection = 0
            reversalVotes = 0
        }
        guard let fused = fused else {
            return held("剖面混杂，暂无可判定的路面结构 · " + motionNote, confidence: 0.30, distance: distance)
        }
        if let step = fused.stepHeight {
            lastStepHeight = step
            lastTreadCount = fused.treadCount
        }
        degradedFrames = 0
        return stabilize(fused)
    }

    /// Keeps the previous verdict for a short grace period instead of
    /// announcing "路况待确认" on every isolated bad frame.
    private func held(_ reason: String, confidence: Float, distance: Float?) -> TerrainResult {
        degradedFrames += 1
        let previous = lastStableKind
        let canHold = previous != .unknown && previous != .dropOff && previous != .obstacle
        if canHold && degradedFrames <= 8 {
            let onFlight = previous == .stairsUp || previous == .stairsDown
            return stabilize(TerrainResult(
                kind: previous,
                confidence: max(0.45, confidence),
                debug: reason + " · 延续上一判定",
                distance: distance,
                stepHeight: onFlight ? lastStepHeight : nil,
                treadCount: onFlight ? lastTreadCount : 0
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

    /// Where the walking surface steps up: the adjacent bin pair with the
    /// largest height jump, and how far away its near side is.
    private func riserFrontier(_ bins: [Bin]) -> (jump: Float, distance: Float)? {
        var bestJump: Float = 0
        var bestDistance: Float?
        for (near, far) in zip(bins, bins.dropFirst()) {
            let jump = abs(far.height - near.height)
            if jump > bestJump {
                bestJump = jump
                bestDistance = near.distance
            }
        }
        guard let distance = bestDistance else { return nil }
        return (jump: bestJump, distance: distance)
    }

    // MARK: - Fused motion

    /// Turns the barometer into a slope that does not care where the camera
    /// points: relative altitude divided by the horizontal distance ARKit says
    /// the phone actually travelled.
    ///
    /// This is the fix for "下楼梯被识别成上楼梯". A wall, a door or a stairwell
    /// at the far end of a flight can tilt the depth profile, but it cannot
    /// lift the phone. If the phone is going down, the barometer says so.
    private func updateMotion(frame: ARFrame, motion: MotionSnapshot) {
        let now = frame.timestamp
        let column = frame.camera.transform.columns.3
        let horizontal = SIMD2<Float>(column.x, column.z)

        while let first = motionTrail.first, now - first.time > 2.9 {
            motionTrail.removeFirst()
        }

        guard let altitude = motion.altitude, motion.altitudeSampleCount >= 3 else {
            if now - motionSlopeUpdatedAt > 1.0 {
                motionSlopeValid = false
                motionSlopeStreak = 0
                motionNote = "运动 气压计不可用"
            }
            return
        }

        if let last = motionTrail.last {
            if now - last.time >= 0.08 {
                motionTrail.append(MotionSample(time: now, horizontal: horizontal, altitude: altitude))
            }
        } else {
            motionTrail.append(MotionSample(time: now, horizontal: horizontal, altitude: altitude))
        }

        guard let first = motionTrail.first, let last = motionTrail.last else { return }
        let span = last.time - first.time

        // Estimate travel by projecting hand positions on the dominant walking
        // axis, then fit altitude against that projection. Using only the two
        // endpoints would make the natural sway of a hand-held phone fake a
        // short (or zero) distance and destroy the slope.
        var mean = SIMD2<Float>(repeating: 0)
        for sample in motionTrail { mean += sample.horizontal }
        mean /= Float(motionTrail.count)

        var sxx: Float = 0
        var sxy: Float = 0
        var syy: Float = 0
        for sample in motionTrail {
            let delta = sample.horizontal - mean
            sxx += delta.x * delta.x
            sxy += delta.x * delta.y
            syy += delta.y * delta.y
        }
        let theta = 0.5 * atan2(2 * sxy, sxx - syy)
        let axis = SIMD2<Float>(cos(theta), sin(theta))

        var minProjection = Float.greatestFiniteMagnitude
        var maxProjection = -Float.greatestFiniteMagnitude
        var sumS: Float = 0
        var sumH: Float = 0
        var sumSS: Float = 0
        var sumSH: Float = 0
        for sample in motionTrail {
            let projection = simd_dot(sample.horizontal - mean, axis)
            let height = Float(sample.altitude)
            minProjection = min(minProjection, projection)
            maxProjection = max(maxProjection, projection)
            sumS += projection
            sumH += height
            sumSS += projection * projection
            sumSH += projection * height
        }
        let travelled = maxProjection - minProjection
        let count = Float(motionTrail.count)
        let denominator = count * sumSS - sumS * sumS
        let fitted = denominator > 1e-4 ? (count * sumSH - sumS * sumH) / denominator : nil

        // Too short a window, standing still, or an implausible jump (tracking
        // reset): no verdict this frame, and the previous one expires quickly.
        if span >= 1.7, travelled >= 0.35, motionTrail.count >= 8,
           let slope = fitted, abs(slope) <= 1.8 {
            motionSlope = motionSlopeValid ? motionSlope * 0.5 + slope * 0.5 : slope
            motionSlopeValid = true
            motionSlopeUpdatedAt = now
            motionSlopeStreak = min(motionSlopeStreak + 1, 60)
        } else if now - motionSlopeUpdatedAt > 1.0 {
            motionSlopeValid = false
            motionSlopeStreak = 0
            motionSlope *= 0.6
        }

        if motionSlopeValid {
            let degrees = atan(motionSlope) * 180.0 / .pi
            let rising = motionSlope > 0.03 ? "上行" : (motionSlope < -0.03 ? "下行" : "近水平")
            motionNote = String(
                format: "运动 %@ %+.0f° · 步频 %.1f",
                rising, degrees, motion.stepRate
            )
        }
    }

    private func directionKind(from candidate: TerrainResult?, sign: Float) -> TerrainKind {
        let isStair = candidate?.kind == .stairsUp || candidate?.kind == .stairsDown
        if isStair { return sign > 0 ? .stairsUp : .stairsDown }
        return sign > 0 ? .uphill : .downhill
    }

    /// Second opinion from the phone's own vertical travel.
    ///
    /// The barometer cannot tell stairs from a ramp, so it never changes the
    /// *kind* of the surface it only overrules the *direction*, and it can turn
    /// a blind "flat" into a slope. A correction is latched so the two signals
    /// cannot flip-flop every few frames.
    private func fuseMotion(
        raw: TerrainResult?,
        timestamp: TimeInterval,
        distance: Float?
    ) -> TerrainResult? {
        let degrees = atan(motionSlope) * 180.0 / .pi
        let strength = abs(degrees)
        let sign: Float = motionSlope > 0 ? 1 : (motionSlope < 0 ? -1 : 0)
        let usable = motionSlopeValid && motionSlopeStreak >= 3 && strength >= 7.0 && sign != 0

        guard usable else {
            if timestamp > motionOverrideUntil { motionOverrideSign = 0 }
            motionConflictFrames = max(0, motionConflictFrames - 1)
            return raw
        }

        // Safety states pass straight through. A latched direction must never
        // be allowed to inherit them — silently turning an obstacle into a
        // "上坡" is exactly the kind of conversion that gets someone hurt.
        if let raw = raw {
            if raw.kind == .obstacle || raw.kind == .dropOff { return raw }
            // A flat reading means the surface levelled out, e.g. a landing.
            // Holding the old heading here is what made reversals feel late.
            if raw.kind == .flat && motionSlopeStreak < 8 { return raw }
        }

        // A latched correction keeps winning while the barometer still agrees,
        // so a stubborn depth profile cannot pull the verdict back and forth.
        if motionOverrideSign == sign && timestamp <= motionOverrideUntil {
            // ...unless the camera now clearly points the other way. The
            // barometer averages the last ~3 s, so immediately after a turn it
            // still reports the old direction; a strong visual reversal is
            // fresher and has to win, otherwise the latch suppresses the whole
            // reversal until it expires.
            if let raw = raw, visualSign(of: raw) == -sign, raw.confidence >= 0.80 {
                motionOverrideSign = 0
                motionOverrideUntil = 0
                motionConflictFrames = 0
                return raw
            }
            stairDirection = sign
            reversalVotes = 0
            let base = raw
            return TerrainResult(
                kind: directionKind(from: base, sign: sign),
                confidence: max(0.52, min(0.78, base?.confidence ?? 0.52)),
                debug: (base?.debug ?? "视觉剖面无结论") + " · 运动主导 \(Int(strength))°",
                distance: base?.distance ?? distance,
                stepHeight: base?.stepHeight,
                treadCount: base?.treadCount ?? 0
            )
        }

        if let candidate = raw {
            let visualSign = self.visualSign(of: candidate)

            if visualSign == sign {
                motionConflictFrames = 0
                motionOverrideSign = 0
                return TerrainResult(
                    kind: candidate.kind,
                    confidence: min(0.95, candidate.confidence + 0.06),
                    debug: candidate.debug + " · " + motionNote + " 一致",
                    distance: candidate.distance,
                    stepHeight: candidate.stepHeight,
                    treadCount: candidate.treadCount
                )
            }

            if visualSign == 0 {
                // Flat profile, yet the phone keeps climbing or descending.
                if motionSlopeStreak >= 8 {
                    motionOverrideSign = sign
                    motionOverrideUntil = timestamp + 2.0
                    return TerrainResult(
                        kind: directionKind(from: candidate, sign: sign),
                        confidence: 0.55,
                        debug: "剖面偏平但高度持续变化 · " + motionNote,
                        distance: candidate.distance ?? distance
                    )
                }
                return candidate
            }

            // Direct conflict. Keep the geometry verdict for a moment, but if
            // the barometer keeps insisting, it wins: it is measuring the
            // actual body movement, not a wall 4 m away.
            //
            // Except when the camera is very sure. A confident visual reading
            // is fresh, while the barometer lags by its averaging window, so
            // right after a turn it must not be overruled.
            motionConflictFrames += 1
            if motionConflictFrames >= 8, candidate.confidence < 0.85 {
                motionConflictFrames = 0
                motionOverrideSign = sign
                motionOverrideUntil = timestamp + 2.0
                stairDirection = sign
                reversalVotes = 0
                return TerrainResult(
                    kind: directionKind(from: candidate, sign: sign),
                    confidence: max(0.55, min(0.80, candidate.confidence - 0.05)),
                    debug: candidate.debug + " · 运动纠正 \(Int(strength))°（原判 \(candidate.kind.rawValue)）",
                    distance: candidate.distance,
                    stepHeight: candidate.stepHeight,
                    treadCount: candidate.treadCount
                )
            }
            return candidate
        }

        // Vision produced nothing at all. A sustained slope is still worth
        // saying out loud instead of leaving the user at "路况待确认".
        if motionSlopeStreak >= 8 {
            motionOverrideSign = sign
            motionOverrideUntil = timestamp + 2.0
            return TerrainResult(
                kind: directionKind(from: nil, sign: sign),
                confidence: 0.52,
                debug: "仅运动判坡 · " + motionNote,
                distance: distance
            )
        }
        return nil
    }

    /// +1 climbing, -1 descending, 0 when the verdict carries no direction.
    private func visualSign(of result: TerrainResult?) -> Float {
        guard let result = result else { return 0 }
        switch result.kind {
        case .stairsUp, .uphill: return 1
        case .stairsDown, .downhill: return -1
        default: return 0
        }
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
        // Only obstacles inside 0.5 m are announced. Anything further away is
        // ordinary room geometry — corridor walls, door frames, furniture —
        // and at that range it is not yet something to stop for. The last bin
        // ends exactly at the limit, so nothing beyond it can produce a verdict.
        let binWidth: Float = 0.06
        let firstDistance: Float = 0.20
        let limitDistance: Float = 0.50
        let binCount = Int(((limitDistance - firstDistance) / binWidth).rounded())
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
                guard forward >= firstDistance, forward <= limitDistance else { continue }
                let index = min(binCount - 1, Int((forward - firstDistance) / binWidth))
                binForward[index].append(forward)
                binHeight[index].append(height)
                binColumns[index] |= 1 << column
            }
        }

        var best: ObstacleReport?
        for index in 0..<binCount {
            let heights = binHeight[index]
            // A wall is seen by several pixels across at least two columns; a
            // speckle or a single railing bar is not enough. A 6 cm slice this
            // close holds plenty of samples when something really is there.
            guard heights.count >= 6, binColumns[index].nonzeroBitCount >= 2 else { continue }
            let sorted = heights.sorted()
            let lowIndex = min(sorted.count - 1, Int(Float(sorted.count) * 0.10))
            let highIndex = min(sorted.count - 1, Int(Float(sorted.count) * 0.90))
            let span = sorted[highIndex] - sorted[lowIndex]
            guard span >= 0.30 else { continue }
            guard let distance = median(binForward[index]) else { continue }

            if onStairs {
                // The riser of the very next step is a vertical surface 17 cm
                // tall sitting right where the sensor is looking. Demand a
                // taller face so stair treads never turn into obstacle alarms,
                // and never suppress a genuinely blocking object.
                guard span >= 0.38 else { continue }
            } else if lastCameraBands >= 2 && span < 0.45 {
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
        let clearWindow = history.suffix(8)
        if clearWindow.count == 8 && clearWindow.allSatisfy({ $0.kind == .unknown }) {
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
        // A reversal used to need five frames *on top of* the gate's 1.6 s and
        // the direction latch's six votes, which stacked into two and a half
        // seconds. The gate is now the single place that prices a reversal, so
        // this layer only needs a bare majority.
        let requiredCount = candidate == .dropOff ? 2 : 3
        let candidateResults = window.filter { $0.kind == candidate }
        let meanConfidence = candidateResults.reduce(Float(0)) { $0 + $1.confidence } / Float(max(1, candidateResults.count))

        guard count >= requiredCount, share >= 0.45, meanConfidence >= 0.40 else {
            // While waiting for confirmation, keep showing the last stable
            // verdict rather than the placeholder.
            let shown: TerrainKind = lastStableKind == .unknown ? .unknown : lastStableKind
            return TerrainResult(
                kind: shown,
                confidence: meanConfidence,
                debug: raw.debug + " · 正在确认",
                distance: raw.distance,
                stepHeight: raw.stepHeight,
                treadCount: raw.treadCount,
                tallStep: shown == .obstacle ? raw.tallStep : nil
            )
        }
        lastStableKind = candidate
        return TerrainResult(
            kind: candidate,
            confidence: meanConfidence,
            debug: raw.debug + " · 稳定 \(count) 帧",
            distance: raw.distance,
            stepHeight: raw.stepHeight,
            treadCount: raw.treadCount,
            tallStep: candidate == .obstacle ? raw.tallStep : nil
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
