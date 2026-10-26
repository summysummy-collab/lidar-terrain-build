import CoreMotion
import Foundation

/// A single fused snapshot of the phone's own motion, used as an independent
/// second opinion next to the LiDAR terrain profile.
///
/// The barometer is the important part: vertical speed divided by the forward
/// speed measured by ARKit gives an effective slope that does not care which
/// surface the camera happens to be pointing at, which is exactly what the
/// LiDAR profile gets wrong when a wall sits at the end of a staircase.
struct MotionSnapshot {
    /// Metres per second, positive when the phone is rising.
    let verticalSpeed: Float
    /// Steps detected per second (0 when standing still).
    let stepRate: Float
    /// Device pitch in degrees, positive when the top of the phone tips up.
    let pitchDegrees: Float
    /// True while the user appears to be walking.
    let isMoving: Bool
    /// False until enough barometer samples exist to trust verticalSpeed.
    let altitudeValid: Bool
    /// Latest relative altitude in metres, nil when no barometer is available.
    let altitude: Double?
    /// Number of barometer samples collected so far.
    let altitudeSampleCount: Int

    static let unavailable = MotionSnapshot(
        verticalSpeed: 0,
        stepRate: 0,
        pitchDegrees: 0,
        isMoving: false,
        altitudeValid: false,
        altitude: nil,
        altitudeSampleCount: 0
    )
}

final class MotionContext {
    private let motionManager = CMMotionManager()
    private let altimeter = CMAltimeter()
    private let queue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "com.walkassist.motion"
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInteractive
        return queue
    }()
    private let lock = NSLock()

    private var altitudeSamples: [(time: TimeInterval, altitude: Double)] = []
    private var stepIntervals: [TimeInterval] = []
    private var lastStepTime: TimeInterval = 0
    private var peakArmed = true
    private var pitchRadians: Double = 0
    private var altitudeAvailable = false

    var isAvailable: Bool {
        motionManager.isDeviceMotionAvailable || CMAltimeter.isRelativeAltitudeAvailable()
    }

    func start() {
        lock.lock()
        altitudeSamples.removeAll()
        stepIntervals.removeAll()
        lastStepTime = 0
        peakArmed = true
        altitudeAvailable = false
        lock.unlock()

        startAltimeter()
        startDeviceMotion()
    }

    func stop() {
        if altimeter.isRelativeAltitudeAvailable() {
            altimeter.stopRelativeAltitudeUpdates()
        }
        if motionManager.isDeviceMotionActive {
            motionManager.stopDeviceMotionUpdates()
        }
        lock.lock()
        altitudeSamples.removeAll()
        stepIntervals.removeAll()
        lock.unlock()
    }

    private func startAltimeter() {
        guard CMAltimeter.isRelativeAltitudeAvailable() else { return }
        altimeter.startRelativeAltitudeUpdates(to: queue) { [weak self] data, _ in
            guard let self = self, let data = data else { return }
            self.lock.lock()
            self.altitudeSamples.append((data.timestamp, data.relativeAltitude.doubleValue))
            if self.altitudeSamples.count > 80 {
                self.altitudeSamples.removeFirst(self.altitudeSamples.count - 80)
            }
            self.altitudeAvailable = true
            self.lock.unlock()
        }
    }

    private func startDeviceMotion() {
        guard motionManager.isDeviceMotionAvailable else { return }
        motionManager.deviceMotionUpdateInterval = 0.02
        motionManager.startDeviceMotionUpdates(to: queue) { [weak self] motion, _ in
            guard let self = self, let motion = motion else { return }
            let acceleration = motion.userAcceleration
            let gravity = motion.gravity
            // Project user acceleration on the gravity axis: a footfall shows
            // up as a sharp vertical peak, horizontal sway does not.
            let vertical = acceleration.x * gravity.x
                + acceleration.y * gravity.y
                + acceleration.z * gravity.z

            self.lock.lock()
            self.pitchRadians = motion.attitude.pitch
            let now = motion.timestamp
            if vertical > 0.11 && self.peakArmed {
                self.peakArmed = false
                if self.lastStepTime > 0 {
                    let interval = now - self.lastStepTime
                    if interval > 0.2 && interval < 2.0 {
                        self.stepIntervals.append(interval)
                        if self.stepIntervals.count > 8 {
                            self.stepIntervals.removeFirst()
                        }
                    }
                }
                self.lastStepTime = now
            } else if vertical < 0.03 {
                self.peakArmed = true
            }
            self.lock.unlock()
        }
    }

    func snapshot() -> MotionSnapshot {
        lock.lock()
        defer { lock.unlock() }

        let now = ProcessInfo.processInfo.systemUptime
        let recent = altitudeSamples.filter { now - $0.time <= 2.2 }

        var verticalSpeed: Double = 0
        var altitudeValid = false
        if recent.count >= 5, let first = recent.first, let last = recent.last {
            let span = last.time - first.time
            if span >= 0.8 {
                // Least-squares slope of altitude over time, in m/s.
                let base = first.time
                var sumT = 0.0
                var sumA = 0.0
                var sumTT = 0.0
                var sumTA = 0.0
                for sample in recent {
                    let t = sample.time - base
                    sumT += t
                    sumA += sample.altitude
                    sumTT += t * t
                    sumTA += t * sample.altitude
                }
                let count = Double(recent.count)
                let denominator = count * sumTT - sumT * sumT
                if abs(denominator) > 1e-9 {
                    verticalSpeed = (count * sumTA - sumT * sumA) / denominator
                    altitudeValid = span >= 1.0 && altitudeAvailable
                }
            }
        }

        let stepRate: Double
        if stepIntervals.isEmpty {
            stepRate = 0
        } else {
            let total = stepIntervals.reduce(0, +)
            stepRate = Double(stepIntervals.count) / max(0.5, total)
        }
        let moving = (now - lastStepTime) < 1.5 && stepRate > 0.6

        return MotionSnapshot(
            verticalSpeed: Float(verticalSpeed),
            stepRate: Float(stepRate),
            pitchDegrees: Float(pitchRadians * 180.0 / .pi),
            isMoving: moving,
            altitudeValid: altitudeValid,
            altitude: altitudeSamples.last?.altitude,
            altitudeSampleCount: altitudeSamples.count
        )
    }
}
