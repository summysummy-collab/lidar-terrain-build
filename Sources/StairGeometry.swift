import Foundation

/// One measured point of the walking surface, in world coordinates relative to
/// the phone: how far ahead, and how far above / below the lens.
struct TerrainPoint {
    let forward: Float
    let height: Float
}

/// What the depth cloud says about the *shape* of the surface ahead.
///
/// The idea this type encodes: **a staircase is discrete, a ramp is
/// continuous.** Treads are horizontal slabs, so in a height histogram the
/// points pile up on a handful of levels separated by exactly the riser
/// height. A ramp spreads the same points evenly across the whole range.
///
/// That difference is far more reliable than asking "how jagged is the fitted
/// line", which is what used to confuse ramps with stairs: a smooth incline
/// with a couple of paving joints in it produced enough residual and enough
/// jumps to look like a staircase, while a real flight seen at an angle could
/// miss the jump threshold and be announced as a ramp.
///
/// It also gives the step height for free, which is what the walking aid needs
/// to tell the user (and the platform) what it is about to climb.
struct StairProfile {
    /// Height of one step in metres, when a staircase was recognised.
    let stepHeight: Float?
    /// Number of distinct horizontal levels found (treads seen).
    let treadCount: Int
    /// Fraction of points that sit on one of those levels. High on stairs,
    /// low on a ramp.
    let discreteness: Float
    /// True when the levels are evenly spaced by a plausible riser height.
    let isLikelyStairs: Bool
    /// 0...1 confidence in the staircase classification.
    let confidence: Float
    /// Compact summary for the on-screen diagnostics.
    let debug: String

    static let none = StairProfile(
        stepHeight: nil,
        treadCount: 0,
        discreteness: 0,
        isLikelyStairs: false,
        confidence: 0,
        debug: "阶梯测量 无样本"
    )
}

enum StairGeometryProbe {
    private static let binSize: Float = 0.025
    private static let minHeight: Float = -1.60
    private static let maxHeight: Float = 1.60
    private static let maxForward: Float = 3.20
    private static let minimumPoints = 24
    private static let minimumRiser: Float = 0.07
    private static let maximumRiser: Float = 0.35
    /// Half-width of the "this point lies on a tread" window.
    private static let treadTolerance: Float = 0.04
    /// Tread levels closer than this are the same tread seen with noise.
    private static let mergeDistance = 4

    static func measure(points: [TerrainPoint]) -> StairProfile {
        let usable = points.filter {
            $0.forward >= 0.45 && $0.forward <= maxForward
                && $0.height >= minHeight && $0.height <= maxHeight
        }
        guard usable.count >= minimumPoints else { return .none }

        let binCount = Int(((maxHeight - minHeight) / binSize).rounded())
        guard binCount > 8 else { return .none }
        var histogram = [Float](repeating: 0, count: binCount)
        for point in usable {
            let index = Int((point.height - minHeight) / binSize)
            if index >= 0, index < binCount {
                histogram[index] += 1
            }
        }
        let smooth = smoothed(histogram)

        // A tread has to hold a meaningful share of all points, otherwise it is
        // noise. Six percent of ~120 samples is seven points landing on one
        // 2.5 cm level. A ramp never reaches that: its points are spread across
        // a metre or more of height, so no single level collects many.
        let minCount = max(Float(2), Float(usable.count) * 0.06)
        var peaks: [Int] = []
        for index in 2..<(binCount - 2) where smooth[index] >= minCount {
            if smooth[index] >= smooth[index - 1], smooth[index] >= smooth[index + 1],
               smooth[index] > smooth[index - 2], smooth[index] > smooth[index + 2] {
                peaks.append(index)
            }
        }
        guard !peaks.isEmpty else {
            return StairProfile(
                stepHeight: nil,
                treadCount: 0,
                discreteness: 0,
                isLikelyStairs: false,
                confidence: 0,
                debug: "阶梯测量 无离散水平面"
            )
        }

        // Two peaks a single bin apart are the same tread with noise on it.
        var merged: [Int] = []
        for peak in peaks {
            if let last = merged.last, peak - last < mergeDistance {
                if smooth[peak] > smooth[last] {
                    merged.removeLast()
                    merged.append(peak)
                }
            } else {
                merged.append(peak)
            }
        }
        if merged.count > 6 {
            merged = Array(merged.sorted { smooth[$0] > smooth[$1] }.prefix(6)).sorted()
        }

        let levels = merged.map { refinedHeight($0, smooth) }.sorted()
        let discreteness = coverage(of: usable, levels: levels)

        var gaps: [Float] = []
        for index in 1..<levels.count {
            let gap = levels[index] - levels[index - 1]
            if gap >= minimumRiser, gap <= maximumRiser { gaps.append(gap) }
        }
        guard let stepHeight = medianOf(gaps) else {
            // A single level is flat ground or one ledge, never a staircase.
            return StairProfile(
                stepHeight: nil,
                treadCount: levels.count,
                discreteness: discreteness,
                isLikelyStairs: false,
                confidence: min(0.40, 0.20 + 0.20 * discreteness),
                debug: String(format: "阶梯测量 %d 个水平面 · 离散 %.0f%%", levels.count, discreteness * 100)
            )
        }

        // Evenly spaced steps are the signature of a flight. A ramp that
        // happened to produce two peaks will not repeat the spacing.
        let consistent = gaps.filter { abs($0 - stepHeight) <= 0.045 }.count
        let regularity = Float(consistent) / Float(gaps.count)
        let plausible = stepHeight >= 0.10 && stepHeight <= 0.28
        let isStairs = levels.count >= 2 && plausible
            && regularity >= 0.55 && discreteness >= 0.45
        let confidence = isStairs
            ? min(0.95, 0.45 + 0.30 * regularity
                  + 0.05 * Float(min(4, levels.count - 1))
                  + 0.20 * discreteness)
            : min(0.50, 0.25 + 0.25 * discreteness)
        let debug = String(
            format: "阶梯 %d 级 · 高 %.0fcm · 规整 %.0f%% · 离散 %.0f%%",
            levels.count, stepHeight * 100, regularity * 100, discreteness * 100
        )
        return StairProfile(
            stepHeight: plausible ? stepHeight : nil,
            treadCount: levels.count,
            discreteness: discreteness,
            isLikelyStairs: isStairs,
            confidence: confidence,
            debug: debug
        )
    }

    /// Five-tap binomial smoothing. Keeps single-sample spikes from becoming
    /// "treads" without erasing a real 15 cm riser.
    private static func smoothed(_ values: [Float]) -> [Float] {
        guard values.count > 4 else { return values }
        let kernel: [Float] = [1, 2, 3, 2, 1]
        let total = kernel.reduce(0, +)
        var result = values
        for index in 0..<values.count {
            var sum: Float = 0
            for (offset, weight) in kernel.enumerated() {
                let source = index + offset - 2
                if source >= 0, source < values.count {
                    sum += values[source] * weight
                }
            }
            result[index] = sum / total
        }
        return result
    }

    /// Weighted centre of the peak and its two neighbours, so the step height
    /// is not quantised to the 2.5 cm histogram.
    private static func refinedHeight(_ index: Int, _ smooth: [Float]) -> Float {
        let lower = max(0, index - 1)
        let upper = min(smooth.count - 1, index + 1)
        var weight: Float = 0
        var sum: Float = 0
        for i in lower...upper {
            let w = smooth[i]
            sum += w * (minHeight + (Float(i) + 0.5) * binSize)
            weight += w
        }
        guard weight > 0 else { return minHeight + (Float(index) + 0.5) * binSize }
        return sum / weight
    }

    private static func coverage(of points: [TerrainPoint], levels: [Float]) -> Float {
        guard !points.isEmpty, !levels.isEmpty else { return 0 }
        var hits = 0
        for point in points {
            for level in levels where abs(point.height - level) <= treadTolerance {
                hits += 1
                break
            }
        }
        return Float(hits) / Float(points.count)
    }

    private static func medianOf(_ values: [Float]) -> Float? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count % 2 == 0 {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }
}
