import ARKit
import SceneKit
import UIKit

final class TerrainViewController: UIViewController, ARSessionDelegate {
    private let arView = ARSCNView()
    private let analyzer = DepthTerrainAnalyzer()
    private let speech = TerrainSpeech()

    private let statusLabel = UILabel()
    private let terrainLabel = UILabel()
    private let evidenceLabel = UILabel()
    private let detailLabel = UILabel()
    private let startButton = UIButton(type: .system)
    private let voiceSwitch = UISwitch()
    private let analysisQueue = DispatchQueue(label: "com.walkassist.terrain.analysis", qos: .userInitiated)
    private let stateLock = NSLock()
    private var runningStorage = false
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
        aimingLabel.text = "将中央区域对准前方地面\n手机尽量固定在胸前，避免大幅摆动"
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
        voiceSwitch.translatesAutoresizingMaskIntoConstraints = false
        let voiceLabel = UILabel()
        voiceLabel.text = "语音提醒"
        voiceLabel.textColor = .white
        voiceLabel.font = .systemFont(ofSize: 15, weight: .medium)
        let voiceStack = UIStackView(arrangedSubviews: [voiceLabel, voiceSwitch])
        voiceStack.axis = .horizontal
        voiceStack.spacing = 8
        voiceStack.alignment = .center
        voiceStack.translatesAutoresizingMaskIntoConstraints = false
        bottomPanel.contentView.addSubview(voiceStack)

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
            voiceStack.topAnchor.constraint(equalTo: startButton.bottomAnchor, constant: 8),
            voiceStack.centerXAnchor.constraint(equalTo: bottomPanel.contentView.centerXAnchor),
            warningLabel.topAnchor.constraint(equalTo: voiceStack.bottomAnchor, constant: 6),
            warningLabel.leadingAnchor.constraint(equalTo: bottomPanel.contentView.leadingAnchor, constant: 8),
            warningLabel.trailingAnchor.constraint(equalTo: bottomPanel.contentView.trailingAnchor, constant: -8),
            warningLabel.bottomAnchor.constraint(equalTo: bottomPanel.contentView.bottomAnchor, constant: -10)
        ])
    }

    private func configure(_ label: UILabel, size: CGFloat, weight: UIFont.Weight) {
        label.font = .systemFont(ofSize: size, weight: weight)
        label.textColor = .white
        label.textAlignment = .center
    }

    @objc private func voiceSwitchChanged() {
        if !voiceSwitch.isOn { speech.stop() }
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
        analysisQueue.sync {
            analyzer.reset()
            lastAnalysisTimestamp = 0
        }
        lastAnnouncedKind = .unknown
        lastAnnouncementAt = .distantPast
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
        analysisQueue.sync {
            analyzer.reset()
            lastAnalysisTimestamp = 0
        }
        speech.stop()
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
            let result = self.analyzer.analyze(frame: frame, displayTransform: displayTransform)
                ?? self.analyzer.unavailable("连续深度帧不可用，请检查镜头遮挡并缓慢移动")
            DispatchQueue.main.async { [weak self] in
                self?.apply(result)
            }
        }
    }

    private func apply(_ result: TerrainResult) {
        guard running else { return }
        var detail = result.debug
        if let distance = result.distance {
            detail += " · 中位距离 \(String(format: "%.1f", distance))m"
        }
        updateTerrain(result.kind, confidence: result.confidence, detail: detail)
        statusLabel.text = "LiDAR + 摄像头融合 · 约 9–10 次/秒"

        guard result.kind != .unknown, result.confidence >= 0.45 else { return }
        let now = Date()
        let changed = result.kind != lastAnnouncedKind
        let urgentRepeat = result.kind == .dropOff && now.timeIntervalSince(lastAnnouncementAt) >= 6
        let periodicReminder = result.kind != .flat && now.timeIntervalSince(lastAnnouncementAt) >= 15
        if changed || urgentRepeat || periodicReminder {
            if voiceSwitch.isOn {
                speech.speak(result.announcement, urgent: result.kind == .dropOff)
            }
            if result.kind == .dropOff {
                UINotificationFeedbackGenerator().notificationOccurred(.warning)
            }
            lastAnnouncedKind = result.kind
            lastAnnouncementAt = now
        }
    }

    private func updateTerrain(_ kind: TerrainKind, confidence: Float, detail: String) {
        terrainLabel.text = kind.rawValue
        evidenceLabel.text = kind == .unknown ? "证据不足时不强行判断方向" : "几何证据强度 \(Int(confidence * 100))%"
        detailLabel.text = detail
        switch kind {
        case .unknown: terrainLabel.textColor = .systemGray
        case .flat: terrainLabel.textColor = .systemGreen
        case .uphill, .downhill, .stairsUp, .stairsDown: terrainLabel.textColor = .systemYellow
        case .dropOff: terrainLabel.textColor = .systemRed
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self, self.running else { return }
            switch camera.trackingState {
            case .normal:
                self.statusLabel.text = "LiDAR 本机实时分析 · 跟踪正常"
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
