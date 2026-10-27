import ARKit
import SceneKit
import UIKit

final class TerrainViewController: UIViewController, ARSessionDelegate {
    private let arView = ARSCNView()
    private let analyzer = DepthTerrainAnalyzer()
    private let speech = TerrainSpeech()
    private let motion = MotionContext()

    private let statusLabel = UILabel()
    private let terrainLabel = UILabel()
    private let evidenceLabel = UILabel()
    private let detailLabel = UILabel()
    private let depthOverlay = UIImageView()
    private let legendLabel = UILabel()
    private let startButton = UIButton(type: .system)
    private let voiceSwitch = UISwitch()
    private let depthSwitch = UISwitch()
    private let analysisQueue = DispatchQueue(label: "com.walkassist.terrain.analysis", qos: .userInitiated)
    private let stateLock = NSLock()
    private var runningStorage = false
    /// Only touched on analysisQueue so the render loop can read it safely.
    private var overlayEnabled = false
    private var overlayTick = 0
    private var running: Bool {
        get {
            stateLock.lock()
            defer { stateLock.unlock() }
            return runningStorage
        }
        set {
            stateLock.lock()
            runningStorage = newValue
            stateLock.unlock()
        }
    }
    private var viewportSize = UIScreen.main.bounds.size
    private var lastAnalysisTimestamp: TimeInterval = 0
    private var lastAnnouncedKind: TerrainKind = .unknown
    private var lastAnnouncementAt = Date.distantPast
    private var lastAnnouncedDistance: Float = 0

    override var preferredStatusBarStyle: UIStatusBarStyle { .lightContent }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        arView.contentMode = .scaleAspectFill
        setupInterface()
        arView.session.delegate = self
        arView.preferredFramesPerSecond = 30
        updateTerrain(.unknown, confidence: 0, detail: "点击开始，允许摄像头权限后使用")
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(applicationDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        viewportSize = arView.bounds.size
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    @objc private func applicationDidEnterBackground() {
        if running { stopSession() }
    }

    private func setupInterface() {
        arView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(arView)
        NSLayoutConstraint.activate([
            arView.topAnchor.constraint(equalTo: view.topAnchor),
            arView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            arView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            arView.trailingAnchor.constraint(equalTo: view.trailingAnchor)
        ])

        depthOverlay.translatesAutoresizingMaskIntoConstraints = false
        depthOverlay.contentMode = .scaleToFill
        depthOverlay.isUserInteractionEnabled = false
        depthOverlay.isHidden = true
        depthOverlay.alpha = 0.55
        view.addSubview(depthOverlay)
        NSLayoutConstraint.activate([
            depthOverlay.topAnchor.constraint(equalTo: arView.topAnchor),
            depthOverlay.bottomAnchor.constraint(equalTo: arView.bottomAnchor),
            depthOverlay.leadingAnchor.constraint(equalTo: arView.leadingAnchor),
            depthOverlay.trailingAnchor.constraint(equalTo: arView.trailingAnchor)
        ])

        let panel = UIVisualEffectView(effect: UIBlurEffect(style: .systemMaterialDark))
        panel.translatesAutoresizingMaskIntoConstraints = false
        panel.layer.cornerRadius = 18
        panel.layer.masksToBounds = true
        view.addSubview(panel)

        let stack = UIStackView()
        stack.axis = .vertical
        stack.spacing = 7
        stack.translatesAutoresizingMaskIntoConstraints = false
        panel.contentView.addSubview(stack)

        configure(statusLabel, size: 13, weight: .regular)
        statusLabel.text = "LiDAR 未启动"
        configure(terrainLabel, size: 34, weight: .bold)
        configure(evidenceLabel, size: 15, weight: .semibold)
        configure(detailLabel, size: 12, weight: .regular)
        detailLabel.numberOfLines = 3
        detailLabel.textColor = UIColor.white.withAlphaComponent(0.72)
        [statusLabel, terrainLabel, evidenceLabel, detailLabel].forEach { stack.addArrangedSubview($0) }

        NSLayoutConstraint.activate([
            panel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            panel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 14),
            panel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -14),
            stack.topAnchor.constraint(equalTo: panel.contentView.topAnchor, constant: 14),
            stack.bottomAnchor.constraint(equalTo: panel.contentView.bottomAnchor, constant: -14),
            stack.leadingAnchor.constraint(equalTo: panel.contentView.leadingAnchor, constant: 16),
            stack.trailingAnchor.constraint(equalTo: panel.contentView.trailingAnchor, constant: -16)
        ])

        let aimingLabel = UILabel()
        aimingLabel.text = "将中央区域对准前方地面\n手机固定在胸前，避免大幅摆动"
        aimingLabel.textAlignment = .center
        aimingLabel.numberOfLines = 2
        aimingLabel.font = .systemFont(ofSize: 13, weight: .medium)
        aimingLabel.textColor = UIColor.white.withAlphaComponent(0.85)
        aimingLabel.layer.shadowColor = UIColor.black.cgColor
        aimingLabel.layer.shadowOpacity = 0.8
        aimingLabel.layer.shadowOffset = .zero
        aimingLabel.layer.shadowRadius = 4
        aimingLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(aimingLabel)
        NSLayoutConstraint.activate([
            aimingLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            aimingLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: 20),
            aimingLabel.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 24),
            aimingLabel.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -24)
        ])

        let bottomPanel = UIVisualEffectView(effect: UIBlurEffect(style: .systemMaterialDark))
        bottomPanel.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.layer.cornerRadius = 18
        bottomPanel.layer.masksToBounds = true
        view.addSubview(bottomPanel)

        startButton.setTitle("开始识别", for: .normal)
        startButton.setTitleColor(.white, for: .normal)
        startButton.titleLabel?.font = .systemFont(ofSize: 19, weight: .bold)
        startButton.backgroundColor = UIColor(red: 0.12, green: 0.48, blue: 0.95, alpha: 1)
        startButton.layer.cornerRadius = 13
        startButton.addTarget(self, action: #selector(toggleRunning), for: .touchUpInside)
        startButton.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.contentView.addSubview(startButton)

        voiceSwitch.isOn = true
        voiceSwitch.addTarget(self, action: #selector(voiceSwitchChanged), for: .valueChanged)
        depthSwitch.isOn = false
        depthSwitch.addTarget(self, action: #selector(depthSwitchChanged), for: .valueChanged)

        let voiceStack = toggleRow(title: "语音提醒", control: voiceSwitch)
        let depthStack = toggleRow(title: "深度叠加", control: depthSwitch)
        let toggleStack = UIStackView(arrangedSubviews: [voiceStack, depthStack])
        toggleStack.axis = .horizontal
        toggleStack.spacing = 20
        toggleStack.alignment = .center
        toggleStack.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.contentView.addSubview(toggleStack)

        legendLabel.text = "叠加色标：红 0.5m · 黄 1.2m · 青 2.4m · 蓝 4m 以上"
        legendLabel.font = .systemFont(ofSize: 11, weight: .regular)
        legendLabel.textColor = UIColor.white.withAlphaComponent(0.7)
        legendLabel.textAlignment = .center
        legendLabel.isHidden = true
        legendLabel.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.contentView.addSubview(legendLabel)

        let warningLabel = UILabel()
        warningLabel.text = "实验性辅助提醒，不能替代观察路面或导盲设施"
        warningLabel.font = .systemFont(ofSize: 11)
        warningLabel.textColor = UIColor.white.withAlphaComponent(0.65)
        warningLabel.textAlignment = .center
        warningLabel.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.contentView.addSubview(warningLabel)

        NSLayoutConstraint.activate([
            aimingLabel.topAnchor.constraint(greaterThanOrEqualTo: panel.bottomAnchor, constant: 12),
            aimingLabel.bottomAnchor.constraint(lessThanOrEqualTo: bottomPanel.topAnchor, constant: -12),
            bottomPanel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 14),
            bottomPanel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -14),
            bottomPanel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -10),
            startButton.topAnchor.constraint(equalTo: bottomPanel.contentView.topAnchor, constant: 12),
            startButton.leadingAnchor.constraint(equalTo: bottomPanel.contentView.leadingAnchor, constant: 14),
            startButton.trailingAnchor.constraint(equalTo: bottomPanel.contentView.trailingAnchor, constant: -14),
            startButton.heightAnchor.constraint(equalToConstant: 52),
            toggleStack.topAnchor.constraint(equalTo: startButton.bottomAnchor, constant: 8),
            toggleStack.centerXAnchor.constraint(equalTo: bottomPanel.contentView.centerXAnchor),
            legendLabel.topAnchor.constraint(equalTo: toggleStack.bottomAnchor, constant: 6),
            legendLabel.leadingAnchor.constraint(equalTo: bottomPanel.contentView.leadingAnchor, constant: 8),
            legendLabel.trailingAnchor.constraint(equalTo: bottomPanel.contentView.trailingAnchor, constant: -8),
            warningLabel.topAnchor.constraint(equalTo: legendLabel.bottomAnchor, constant: 4),
            warningLabel.leadingAnchor.constraint(equalTo: bottomPanel.contentView.leadingAnchor, constant: 8),
            warningLabel.trailingAnchor.constraint(equalTo: bottomPanel.contentView.trailingAnchor, constant: -8),
            warningLabel.bottomAnchor.constraint(equalTo: bottomPanel.contentView.bottomAnchor, constant: -10)
        ])
    }

    private func toggleRow(title: String, control: UISwitch) -> UIStackView {
        let label = UILabel()
        label.text = title
        label.textColor = .white
        label.font = .systemFont(ofSize: 15, weight: .medium)
        let row = UIStackView(arrangedSubviews: [label, control])
        row.axis = .horizontal
        row.spacing = 8
        row.alignment = .center
        return row
    }

    private func configure(_ label: UILabel, size: CGFloat, weight: UIFont.Weight) {
        label.font = .systemFont(ofSize: size, weight: weight)
        label.textColor = .white
        label.textAlignment = .center
    }

    @objc private func voiceSwitchChanged() {
        if !voiceSwitch.isOn { speech.stop() }
    }

    @objc private func depthSwitchChanged() {
        let enabled = depthSwitch.isOn
        analysisQueue.sync {
            overlayEnabled = enabled
            overlayTick = 0
        }
        legendLabel.isHidden = !enabled
        depthOverlay.isHidden = !enabled
        if !enabled { depthOverlay.image = nil }
    }

    @objc private func toggleRunning() {
        if running {
            stopSession()
        } else {
            startSession()
        }
    }

    private func startSession() {
        guard ARWorldTrackingConfiguration.isSupported,
              ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            updateTerrain(.unknown, confidence: 0, detail: "此设备不支持 ARKit LiDAR sceneDepth")
            statusLabel.text = "需要带激光雷达的 iPhone Pro"
            return
        }

        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            configuration.frameSemantics = .smoothedSceneDepth
        } else {
            configuration.frameSemantics = .sceneDepth
        }
        motion.start()
        analysisQueue.sync {
            analyzer.reset()
            lastAnalysisTimestamp = 0
        }
        lastAnnouncedKind = .unknown
        lastAnnouncementAt = .distantPast
        lastAnnouncedDistance = 0
        arView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        running = true
        UIApplication.shared.isIdleTimerDisabled = true
        startButton.setTitle("停止", for: .normal)
        startButton.backgroundColor = UIColor(red: 0.82, green: 0.20, blue: 0.18, alpha: 1)
        statusLabel.text = "LiDAR 正在获取深度…"
        if voiceSwitch.isOn { speech.unlock() }
    }

    private func stopSession() {
        running = false
        arView.session.pause()
        motion.stop()
        analysisQueue.sync {
            analyzer.reset()
            lastAnalysisTimestamp = 0
        }
        speech.stop()
        depthOverlay.image = nil
        UIApplication.shared.isIdleTimerDisabled = false
        startButton.setTitle("开始识别", for: .normal)
        startButton.backgroundColor = UIColor(red: 0.12, green: 0.48, blue: 0.95, alpha: 1)
        statusLabel.text = "已停止"
        updateTerrain(.unknown, confidence: 0, detail: "点击开始重新识别")
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        analysisQueue.async { [weak self] in
            guard let self = self, self.running,
                  frame.timestamp - self.lastAnalysisTimestamp >= 0.10 else { return }
            self.lastAnalysisTimestamp = frame.timestamp
            let displayTransform = frame.displayTransform(for: .portrait, viewportSize: self.viewportSize)
            let snapshot = self.motion.snapshot()
            let result = self.analyzer.analyze(
                frame: frame,
                displayTransform: displayTransform,
                motion: snapshot
            ) ?? self.analyzer.unavailable("连续深度帧不可用，请检查镜头遮挡并缓慢移动")
            var overlay: UIImage?
            if self.overlayEnabled {
                self.overlayTick += 1
                if self.overlayTick % 2 == 1 {
                    overlay = self.makeDepthOverlay(frame: frame, displayTransform: displayTransform)
                }
            }
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                if let overlay = overlay, self.overlayEnabled {
                    self.depthOverlay.image = overlay
                }
                self.apply(result)
            }
        }
    }

    /// Renders the LiDAR depth map as a colour image sampled through the same
    /// display transform the analyzer uses, so the overlay lines up with the
    /// camera picture instead of drifting.
    private func makeDepthOverlay(frame: ARFrame, displayTransform: CGAffineTransform) -> UIImage? {
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return nil }
        let map = depthData.depthMap
        guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
              CVPixelBufferLockBaseAddress(map, .readOnly) == kCVReturnSuccess,
              let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }

        let mapWidth = CVPixelBufferGetWidth(map)
        let mapHeight = CVPixelBufferGetHeight(map)
        let stride = CVPixelBufferGetBytesPerRow(map) / MemoryLayout<Float>.stride
        let pointer = base.assumingMemoryBound(to: Float.self)

        let gridWidth = 72
        let viewportWidth = max(CGFloat(1), viewportSize.width)
        let aspect = max(CGFloat(1), viewportSize.height / viewportWidth)
        let gridHeight = max(16, Int((CGFloat(gridWidth) * aspect).rounded()))
        var bytes = [UInt8](repeating: 0, count: gridWidth * gridHeight * 4)

        for row in 0..<gridHeight {
            let viewV = (CGFloat(row) + 0.5) / CGFloat(gridHeight)
            for column in 0..<gridWidth {
                let viewU = (CGFloat(column) + 0.5) / CGFloat(gridWidth)
                let imagePoint = CGPoint(x: viewU, y: viewV).applying(displayTransform)
                let x = Int(imagePoint.x * CGFloat(mapWidth - 1))
                let y = Int(imagePoint.y * CGFloat(mapHeight - 1))
                guard x >= 0, x < mapWidth, y >= 0, y < mapHeight else { continue }
                let depth = pointer[y * stride + x]
                guard depth.isFinite, depth > 0.1, depth <= 6.0 else { continue }
                let color = TerrainViewController.overlayColor(for: depth)
                let index = (row * gridWidth + column) * 4
                bytes[index] = color.0
                bytes[index + 1] = color.1
                bytes[index + 2] = color.2
                bytes[index + 3] = 255
            }
        }

        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cgImage = CGImage(
                width: gridWidth,
                height: gridHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: gridWidth * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else { return nil }
        return UIImage(cgImage: cgImage)
    }

    /// Red = very close, orange/yellow = walking range, cyan/blue = open space.
    private static func overlayColor(for depth: Float) -> (UInt8, UInt8, UInt8) {
        let stops: [(Float, Float, Float, Float)] = [
            (0.30, 255, 40, 40),
            (0.80, 255, 140, 0),
            (1.40, 255, 220, 40),
            (2.20, 60, 220, 180),
            (3.20, 60, 160, 255),
            (4.50, 90, 90, 255)
        ]
        func channel(_ value: Float) -> UInt8 {
            UInt8(max(0, min(255, value)))
        }
        if depth <= stops[0].0 {
            let first = stops[0]
            return (channel(first.1), channel(first.2), channel(first.3))
        }
        for index in 1..<stops.count where depth <= stops[index].0 {
            let low = stops[index - 1]
            let high = stops[index]
            let span = max(0.01, high.0 - low.0)
            let t = (depth - low.0) / span
            return (
                channel(low.1 + (high.1 - low.1) * t),
                channel(low.2 + (high.2 - low.2) * t),
                channel(low.3 + (high.3 - low.3) * t)
            )
        }
        let last = stops[stops.count - 1]
        return (channel(last.1), channel(last.2), channel(last.3))
    }

    private func apply(_ result: TerrainResult) {
        guard running else { return }
        var detail = result.debug
        if let distance = result.distance {
            detail += " · 中位距离 \(String(format: "%.1f", distance))m"
        }
        if result.heldSeconds >= 0.05 {
            detail += String(format: " · 已稳定 %.1fs", result.heldSeconds)
        }
        updateTerrain(
            result.kind,
            confidence: result.confidence,
            detail: detail,
            pending: result.pendingKind,
            progress: result.pendingProgress
        )
        statusLabel.text = "v6 · 状态门控 · 障碍 1.5m 内 · 9–10 次/秒"

        guard result.kind != .unknown, result.confidence >= 0.45 else { return }
        let now = Date()
        let hazard = result.kind == .dropOff || result.kind == .obstacle
        let obstacle = result.kind == .obstacle ? (result.distance ?? 0) : 0
        let changed = result.kind != lastAnnouncedKind
        // Walking towards an obstacle changes the picture: re-announce as the
        // measured distance drops instead of repeating one stale number.
        let closingIn = result.kind == .obstacle
            && lastAnnouncedKind == .obstacle
            && abs(obstacle - lastAnnouncedDistance) >= 0.4
        let repeatInterval: TimeInterval = result.kind == .dropOff ? 6 : 2
        let urgentRepeat = hazard && now.timeIntervalSince(lastAnnouncementAt) >= repeatInterval
        let periodicReminder = result.kind != .flat && !hazard && now.timeIntervalSince(lastAnnouncementAt) >= 15
        if changed || closingIn || urgentRepeat || periodicReminder {
            if voiceSwitch.isOn {
                speech.speak(result.announcement, urgent: hazard)
            }
            if hazard {
                UINotificationFeedbackGenerator().notificationOccurred(.warning)
            }
            lastAnnouncedKind = result.kind
            lastAnnouncedDistance = obstacle
            lastAnnouncementAt = now
        }
    }

    private func updateTerrain(
        _ kind: TerrainKind,
        confidence: Float,
        detail: String,
        pending: TerrainKind? = nil,
        progress: Float = 0
    ) {
        terrainLabel.text = kind.rawValue
        if let pending = pending {
            evidenceLabel.text = "→ \(pending.rawValue) \(Int(progress * 100))% · 转换中，请减速"
            evidenceLabel.textColor = .systemOrange
        } else {
            evidenceLabel.text = kind == .unknown ? "证据不足时不强行判断方向" : "几何证据强度 \(Int(confidence * 100))%"
            evidenceLabel.textColor = .white
        }
        detailLabel.text = detail
        switch kind {
        case .unknown: terrainLabel.textColor = .systemGray
        case .flat: terrainLabel.textColor = .systemGreen
        case .uphill, .downhill, .stairsUp, .stairsDown: terrainLabel.textColor = .systemYellow
        case .obstacle: terrainLabel.textColor = .systemOrange
        case .dropOff: terrainLabel.textColor = .systemRed
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self, self.running else { return }
            switch camera.trackingState {
            case .normal:
                self.statusLabel.text = "v6 · 状态门控 · 跟踪正常"
            case .notAvailable:
                self.statusLabel.text = "空间跟踪不可用"
            case .limited:
                self.statusLabel.text = "空间跟踪受限，请缓慢移动并对准有纹理的区域"
            }
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            if self.running { self.stopSession() }
            self.statusLabel.text = "AR 会话被中断，返回后请重新开始"
        }
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        DispatchQueue.main.async { [weak self] in
            self?.statusLabel.text = "中断已结束，请点击开始重新识别"
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            if self.running { self.stopSession() }
            self.statusLabel.text = "AR 会话失败：\(error.localizedDescription)"
        }
    }
}
