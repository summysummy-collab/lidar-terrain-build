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
        let required = requiredSeconds(from: settled, to: candidate, evidence: input.confidence) + patience
        // Even a well-evidenced flip waits until the current state has had its
        // minimum airtime, so the platform never gets two commands in a row.
        let dwellWait = max(0, minimumDwell(settled) - held)
        let need = max(required, dwellWait)
        let elapsed = time - candidateSince

        if elapsed >= need {
            adopt(candidate, from: settled, at: time)
            return finish(input, time: time, pending: nil, progress: 0)
        }

        // Publish an early warning before the move is decided, so a moving
        // platform can decelerate while the state is still changing. On the
        // fast path this fires almost immediately, which is the point: a
        // landing turn has to reach the actuator in well under a second.
        let progress = Float(min(1.0, elapsed / max(0.05, need)))
        let pending: TerrainKind? = progress >= 0.35 ? candidate : nil
        return finish(input, time: time, pending: pending, progress: progress)
    }

    private func adopt(_ kind: TerrainKind, from previous: TerrainKind, at time: TimeInterval) {
        // A -> B -> A inside three seconds means the surface is hugging a
        // threshold. Make the next attempt cost more instead of echoing it.
        // Capped so a genuinely changing surface recovers quickly: a person
        // turning round twice on a wide landing is not a sensor fault.
        if lastFlipFrom == kind && lastFlipTo == previous && time - lastFlipAt < 3.0 {
            patience = min(1.5, patience + 0.8)
        } else {
            patience = max(0, patience - 0.6)
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
            treadCount: input.treadCount,
            tallStep: settled == .obstacle ? input.tallStep : nil
        )
    }

    /// How long a move must be supported before it is allowed through.
    ///
    /// The delay is no longer one flat number per transition type — that made
    /// turning round on a landing take two and a half seconds, by which time
    /// the platform had already travelled three metres the wrong way. A
    /// confident measurement now gets through in about a third of a second;
    /// only weak evidence pays the old price, because only weak evidence is
    /// what actually flickers.
    private func requiredSeconds(from: TerrainKind, to: TerrainKind, evidence: Float) -> TimeInterval {
        // Safety pre-empts everything: a drop-off must not wait in a queue.
        if to == .dropOff { return 0.10 }
        if from == .dropOff { return 0.25 }
        if to == .obstacle { return 0.30 }
        if from == .obstacle { return 0.25 }
        // "Don't know" has to be earned: it means data dried up entirely.
        if to == .unknown { return 1.40 }
        if from == .unknown { return 0.30 }

        let fromDirection = direction(of: from)
        let toDirection = direction(of: to)
        if fromDirection != 0 && toDirection != 0 && fromDirection != toDirection {
            // Up vs down is the most expensive confusion to recover from, so it
            // is scaled by how sure the measurement is rather than fixed.
            if evidence >= 0.88 { return 0.35 }
            if evidence >= 0.75 { return 0.55 }
            if evidence >= 0.60 { return 0.85 }
            return 1.25
        }
        if isStair(from) != isStair(to) {
            // Stairs vs ramp: same height change, different gait.
            return 0.55
        }
        if to == .flat || from == .flat {
            return 0.60
        }
        return 0.45
    }

    private func minimumDwell(_ kind: TerrainKind) -> TimeInterval {
        switch kind {
        case .stairsUp, .stairsDown: return 0.55
        case .uphill, .downhill: return 0.45
        case .flat: return 0.35
        case .obstacle: return 0.30
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
