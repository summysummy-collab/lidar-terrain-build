import AVFoundation

final class TerrainSpeech: NSObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    private var lastText = ""
    private var lastSpokenAt = Date.distantPast

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func unlock() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try audioSession.setActive(true)
        } catch {
            // Speech synthesis can still attempt playback with the current audio session.
        }
        speak("激光雷达路况识别已启动", urgent: true, force: true)
    }

    func speak(_ text: String, urgent: Bool = false, force: Bool = false) {
        guard !text.isEmpty else { return }
        let now = Date()
        if !force && text == lastText && now.timeIntervalSince(lastSpokenAt) < (urgent ? 3 : 8) {
            return
        }
        if urgent || synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
        utterance.rate = urgent ? 0.54 : 0.48
        utterance.volume = 1.0
        synthesizer.speak(utterance)
        lastText = text
        lastSpokenAt = now
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        lastText = ""
        lastSpokenAt = .distantPast
    }
}
