import Foundation

/// Turns a per-frame terrain vote into a command a moving platform can follow.
///
/// A walking aid that only talks to a human can afford to be jumpy: the person
/// filters it out. A platform travelling at 1–2 m/s cannot — every flip means
/// start/stop, and at 1–2 m/s a 300 ms wobble already covers half a metre.
///
/// So the raw verdict is passed through here, which enforces three things:
///
///  1. **Minimum dwell.** Once a state is adopted it cannot be left before it
///     has been held for a while, no matter what the next frames say.
///  2. **A cost table.** Reclassifying "上坡 → 上楼梯" (same direction, similar
///     act) is cheap; reversing "上楼梯 → 下楼梯" is expensive. Danger states
///     pre-empt everything.
///  3. **A pending flag.** Halfway to a flip the gate publishes `pendingKind`
///     so the platform can ramp down *before* the verdict changes.
///
/// Ping-pong is punished: an A → B → A round trip within a few seconds adds
/// patience, which is what stops a threshold-hugging surface from oscillating.
final class TerrainStabilityGate {
    private(set) var settled: TerrainKind = .unknown
    private var settledAt: TimeInterval = -1
    private var candidate: TerrainKind?
    private var candidateSince: TimeInterval = 0
    private var lastFlipFrom: TerrainKind = .unknown
    private var lastFlipTo: TerrainKind = .unknown
    private var lastFlipAt: TimeInterval = -100
    /// Extra waiting accumulated from repeated back-and-forth flips.
    private var patience: TimeInterval = 0

    func reset() {
        settled = .unknown
        settledAt = -1
        candidate = nil
        candidateSince = 0
        lastFlipFrom = .unknown
        lastFlipTo = .unknown
        lastFlipAt = -100
        patience = 0
    }

    func process(_ input: TerrainResult, at time: TimeInterval) -> TerrainResult {
        // First frame: adopt immediately, there is nothing to protect yet.
        if settledAt < 0 {
            settled = input.kind
            settledAt = time
            candidate = nil
            return finish(input, time: time, pending: nil, progress: 0)
        }

        if input.kind == settled {
            candidate = nil
            return finish(input, time: time, pending: nil, progress: 0)
        }

        if candidate != input.kind {
            candidate = input.kind
            candidateSince = time
        }
        guard let candidate = candidate else {
            return finish(input, time: time, pending: nil, progress: 0)
        }

        let held = time - settledAt
        let required = requiredSeconds(from: settled, to: candidate) + patience
        // Even a well-evidenced flip waits until the current state has had its
        // minimum airtime, so the platform never gets two commands in a row.
        let dwellWait = max(0, minimumDwell(settled) - held)
        let need = max(required, dwellWait)
        let elapsed = time - candidateSince

        if elapsed >= need {
            adopt(candidate, from: settled, at: time)
            return finish(input, time: time, pending: nil, progress: 0)
        }

        // Publish an early warning once the move is more than half decided, so
        // a moving platform can decelerate before the state actually changes.
        let progress = Float(min(1.0, elapsed / max(0.05, need)))
        let pending: TerrainKind? = progress >= 0.5 ? candidate : nil
        return finish(input, time: time, pending: pending, progress: progress)
    }

    private func adopt(_ kind: TerrainKind, from previous: TerrainKind, at time: TimeInterval) {
        // A -> B -> A inside three seconds means the surface is hugging a
        // threshold. Make the next attempt cost more instead of echoing it.
        if lastFlipFrom == kind && lastFlipTo == previous && time - lastFlipAt < 3.0 {
            patience = min(2.0, patience + 1.0)
        } else {
            patience = max(0, patience - 0.5)
        }
        lastFlipFrom = previous
        lastFlipTo = kind
        lastFlipAt = time
        settled = kind
        settledAt = time
        candidate = nil
    }

    private func finish(
        _ input: TerrainResult,
        time: TimeInterval,
        pending: TerrainKind?,
        progress: Float
    ) -> TerrainResult {
        let held = max(0, time - settledAt)
        // While a flip is pending the confidence of the *held* state is
        // deliberately reported lower: that is the platform's cue to ease off.
        let confidence = pending == nil
            ? input.confidence
            : max(0.35, input.confidence * 0.75)
        return TerrainResult(
            kind: settled,
            confidence: confidence,
            debug: input.debug,
            distance: input.distance,
            heldSeconds: held,
            pendingKind: pending,
            pendingProgress: progress,
            stepHeight: input.stepHeight,
            treadCount: input.treadCount
        )
    }

    /// How long a move must be supported before it is allowed through.
    private func requiredSeconds(from: TerrainKind, to: TerrainKind) -> TimeInterval {
        // Safety pre-empts everything: a drop-off must not wait in a queue.
        if to == .dropOff { return 0.10 }
        if from == .dropOff { return 0.30 }
        if to == .obstacle { return 0.35 }
        if from == .obstacle { return 0.30 }
        // "Don't know" has to be earned: it means data dried up entirely.
        if to == .unknown { return 1.80 }
        if from == .unknown { return 0.45 }

        let fromDirection = direction(of: from)
        let toDirection = direction(of: to)
        if fromDirection != 0 && toDirection != 0 && fromDirection != toDirection {
            // Up vs down is the most expensive confusion to recover from.
            return 1.60
        }
        if isStair(from) != isStair(to) {
            // Stairs vs ramp: same height change, different gait.
            return 0.90
        }
        if to == .flat || from == .flat {
            return 0.90
        }
        return 0.70
    }

    private func minimumDwell(_ kind: TerrainKind) -> TimeInterval {
        switch kind {
        case .stairsUp, .stairsDown: return 1.20
        case .uphill, .downhill: return 0.80
        case .flat: return 0.60
        case .obstacle: return 0.40
        case .dropOff, .unknown: return 0
        }
    }

    private func direction(of kind: TerrainKind) -> Int {
        switch kind {
        case .uphill, .stairsUp: return 1
        case .downhill, .stairsDown: return -1
        default: return 0
        }
    }

    private func isStair(_ kind: TerrainKind) -> Bool {
        kind == .stairsUp || kind == .stairsDown
    }
}
