import AVFoundation
import Combine
import Foundation
import os.log

/// Main class for ElevenLabsSwift package
public class ElevenLabsSDK {
    public static let version = "1.1.3"

    private enum Constants {
        static let defaultApiOrigin = "wss://api.elevenlabs.io"
        static let defaultApiPathname = "/v1/convai/conversation?agent_id="
        static let inputSampleRate: Double = 16000
        static let sampleRate: Double = 16000
        static let ioBufferDuration: Double = 0.005
        static let volumeUpdateInterval: TimeInterval = 0.1
        static let fadeOutDuration: TimeInterval = 2.0
        static let bufferSize: AVAudioFrameCount = 1024

        // WebSocket message size limits
        static let maxWebSocketMessageSize = 1024 * 1024 // 1MB WebSocket limit
        static let safeMessageSize = 750 * 1024 // 750KB - safely under the limit
        static let maxRequestedMessageSize = 8 * 1024 * 1024 // 8MB - request larger buffer if available
    }

    // MARK: - Session Config Utilities

    public enum Language: String, Codable, Sendable {
        case en, ja, zh, de, hi, fr, ko, pt, it, es, id, nl, tr, pl, sv, bg, ro, ar, cs, el, fi, ms, da, ta, uk, ru, hu, no, vi
    }

    public struct AgentPrompt: Codable, Sendable {
        public var prompt: String?

        public init(prompt: String? = nil) {
            self.prompt = prompt
        }
    }

    public struct TTSConfig: Codable, Sendable {
        public var voiceId: String?

        private enum CodingKeys: String, CodingKey {
            case voiceId = "voice_id"
        }

        public init(voiceId: String? = nil) {
            self.voiceId = voiceId
        }
    }

    public struct ConversationConfigOverride: Codable, Sendable {
        public var agent: AgentConfig?
        public var tts: TTSConfig?

        public init(agent: AgentConfig? = nil, tts: TTSConfig? = nil) {
            self.agent = agent
            self.tts = tts
        }
    }

    public struct AgentConfig: Codable, Sendable {
        public var prompt: AgentPrompt?
        public var firstMessage: String?
        public var language: Language?

        private enum CodingKeys: String, CodingKey {
            case prompt
            case firstMessage = "first_message"
            case language
        }

        public init(prompt: AgentPrompt? = nil, firstMessage: String? = nil, language: Language? = nil) {
            self.prompt = prompt
            self.firstMessage = firstMessage
            self.language = language
        }
    }

    public enum LlmExtraBodyValue: Codable, Sendable {
        case string(String)
        case number(Double)
        case boolean(Bool)
        case null
        case array([LlmExtraBodyValue])
        case dictionary([String: LlmExtraBodyValue])

        var jsonValue: Any {
            switch self {
            case let .string(str): return str
            case let .number(num): return num
            case let .boolean(bool): return bool
            case .null: return NSNull()
            case let .array(arr): return arr.map { $0.jsonValue }
            case let .dictionary(dict): return dict.mapValues { $0.jsonValue }
            }
        }
    }

    // MARK: - Audio Utilities

    public static func arrayBufferToBase64(_ data: Data) -> String {
        data.base64EncodedString()
    }

    public static func base64ToArrayBuffer(_ base64: String) -> Data? {
        Data(base64Encoded: base64)
    }

    // MARK: - Client Tools

    public typealias ClientToolHandler = @Sendable (Parameters) async throws -> String?

    public typealias Parameters = [String: Any]

    public struct ClientTools: Sendable {
        private var tools: [String: ClientToolHandler] = [:]
        private let lock = NSLock() // Ensure thread safety

        public init() {}

        public mutating func register(_ name: String, handler: @escaping ClientToolHandler) {
            lock.withLock {
                tools[name] = handler
            }
        }

        public func handle(_ name: String, parameters: Parameters) async throws -> String? {
            let handler: ClientToolHandler? = lock.withLock { tools[name] }
            guard let handler = handler else {
                throw ClientToolError.handlerNotFound(name)
            }
            return try await handler(parameters)
        }
    }

    public enum ClientToolError: Error {
        case handlerNotFound(String)
        case invalidParameters
        case executionFailed(String)
    }

    // MARK: - Connection

    public enum DynamicVariableValue: Sendable {
        case string(String)
        case number(Double)
        case boolean(Bool)
        case int(Int)

        var jsonValue: Any {
            switch self {
            case let .string(str): return str
            case let .number(num): return num
            case let .boolean(bool): return bool
            case let .int(int): return int
            }
        }
    }

    public struct SessionConfig: Sendable {
        public let signedUrl: String?
        public let agentId: String?
        public let overrides: ConversationConfigOverride?
        public let customLlmExtraBody: [String: LlmExtraBodyValue]?
        public let dynamicVariables: [String: DynamicVariableValue]?

        public init(signedUrl: String, overrides: ConversationConfigOverride? = nil, customLlmExtraBody: [String: LlmExtraBodyValue]? = nil, clientTools _: ClientTools = ClientTools(), dynamicVariables: [String: DynamicVariableValue]? = nil) {
            self.signedUrl = signedUrl
            agentId = nil
            self.overrides = overrides
            self.customLlmExtraBody = customLlmExtraBody
            self.dynamicVariables = dynamicVariables
        }

        public init(agentId: String, overrides: ConversationConfigOverride? = nil, customLlmExtraBody: [String: LlmExtraBodyValue]? = nil, clientTools _: ClientTools = ClientTools(), dynamicVariables: [String: DynamicVariableValue]? = nil) {
            self.agentId = agentId
            signedUrl = nil
            self.overrides = overrides
            self.customLlmExtraBody = customLlmExtraBody
            self.dynamicVariables = dynamicVariables
        }
    }

    public class Connection: @unchecked Sendable {
        public let socket: URLSessionWebSocketTask
        public let conversationId: String
        public let sampleRate: Int

        private init(socket: URLSessionWebSocketTask, conversationId: String, sampleRate: Int) {
            self.socket = socket
            self.conversationId = conversationId
            self.sampleRate = sampleRate
        }

        public static func create(config: SessionConfig) async throws -> Connection {
            let origin = ProcessInfo.processInfo.environment["ELEVENLABS_CONVAI_SERVER_ORIGIN"] ?? Constants.defaultApiOrigin
            let pathname = ProcessInfo.processInfo.environment["ELEVENLABS_CONVAI_SERVER_PATHNAME"] ?? Constants.defaultApiPathname

            let urlString: String
            if let signedUrl = config.signedUrl {
                urlString = signedUrl
            } else if let agentId = config.agentId {
                urlString = "\(origin)\(pathname)\(agentId)"
            } else {
                throw ElevenLabsError.invalidConfiguration
            }

            guard let url = URL(string: urlString) else {
                throw ElevenLabsError.invalidURL
            }

            let session = URLSession(configuration: .default)
            let socket = session.webSocketTask(with: url)
            socket.resume()

            // Always send initialization event
            var initEvent: [String: Any] = ["type": "conversation_initiation_client_data"]

            // Add overrides if present
            if let overrides = config.overrides,
               let overridesDict = overrides.dictionary
            {
                initEvent["conversation_config_override"] = overridesDict
            }

            // Add custom body if present
            if let customBody = config.customLlmExtraBody {
                initEvent["custom_llm_extra_body"] = customBody.mapValues { $0.jsonValue }
            }

            // Add dynamic variables if present - Convert to JSON-compatible values
            if let dynamicVars = config.dynamicVariables {
                initEvent["dynamic_variables"] = dynamicVars.mapValues { $0.jsonValue }
            }

            let jsonData = try JSONSerialization.data(withJSONObject: initEvent)
            let jsonString = String(data: jsonData, encoding: .utf8)!
            try await socket.send(.string(jsonString))

            let configData = try await receiveInitialMessage(socket: socket)
            return Connection(socket: socket, conversationId: configData.conversationId, sampleRate: configData.sampleRate)
        }

        private static func receiveInitialMessage(
            socket: URLSessionWebSocketTask
        ) async throws -> (conversationId: String, sampleRate: Int) {
            return try await withCheckedThrowingContinuation { continuation in
                socket.receive { result in
                    switch result {
                    case let .success(message):
                        switch message {
                        case let .string(text):
                            guard let data = text.data(using: .utf8),
                                  let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                                  let type = json["type"] as? String,
                                  type == "conversation_initiation_metadata",
                                  let metadata = json["conversation_initiation_metadata_event"] as? [String: Any],
                                  let conversationId = metadata["conversation_id"] as? String,
                                  let audioFormat = metadata["agent_output_audio_format"] as? String
                            else {
                                continuation.resume(throwing: ElevenLabsError.invalidInitialMessageFormat)
                                return
                            }

                            let sampleRate = Int(audioFormat.replacingOccurrences(of: "pcm_", with: "")) ?? 16000
                            continuation.resume(returning: (conversationId: conversationId, sampleRate: sampleRate))

                        case .data:
                            continuation.resume(throwing: ElevenLabsError.unexpectedBinaryMessage)

                        @unknown default:
                            continuation.resume(throwing: ElevenLabsError.unknownMessageType)
                        }
                    case let .failure(error):
                        continuation.resume(throwing: error)
                    }
                }
            }
        }

        public func close() {
            socket.cancel(with: .goingAway, reason: nil)
        }
    }
    
    // MARK: - Conversation

    public enum Role: String {
        case user
        case ai
    }

    public enum Mode: String {
        case speaking
        case listening
    }

    public enum Status: String {
        case connecting
        case connected
        case disconnecting
        case disconnected
    }

    public struct Callbacks: Sendable {
        public var onConnect: @Sendable (String) -> Void = { _ in }
        public var onDisconnect: @Sendable () -> Void = {}
        public var onMessage: @Sendable (String, Role) -> Void = { _, _ in }
        public var onError: @Sendable (String, Any?) -> Void = { _, _ in }
        public var onAudioEvent: @Sendable (Data) -> Void = { _ in }
        public var onInterruption: @Sendable () -> Void = {}
        public var onStatusChange: @Sendable (Status) -> Void = { _ in }
        public var onModeChange: @Sendable (Mode) -> Void = { _ in }

        public init() {}
    }

    public class Conversation: @unchecked Sendable {
        private let connection: Connection
        private let callbacks: Callbacks
        private let clientTools: ClientTools?

        private let modeLock = NSLock()
        private let statusLock = NSLock()
        private let lastInterruptTimestampLock = NSLock()
        private let isProcessingInputLock = NSLock()

        private var _mode: Mode = .listening
        private var _status: Status = .connecting
        private var _lastInterruptTimestamp: Int = 0
        private var _isProcessingInput: Bool = true

        private var mode: Mode {
            get { modeLock.withLock { _mode } }
            set { modeLock.withLock { _mode = newValue } }
        }

        private var status: Status {
            get { statusLock.withLock { _status } }
            set { statusLock.withLock { _status = newValue } }
        }

        private var lastInterruptTimestamp: Int {
            get { lastInterruptTimestampLock.withLock { _lastInterruptTimestamp } }
            set { lastInterruptTimestampLock.withLock { _lastInterruptTimestamp = newValue } }
        }

        private var isProcessingInput: Bool {
            get { isProcessingInputLock.withLock { _isProcessingInput } }
            set { isProcessingInputLock.withLock { _isProcessingInput = newValue } }
        }

        private let logger = Logger(subsystem: "com.elevenlabs.ElevenLabsSDK", category: "Conversation")

        private init(connection: Connection, callbacks: Callbacks, clientTools: ClientTools?) {
            self.connection = connection
            self.callbacks = callbacks
            self.clientTools = clientTools

            setupWebSocket()
        }

        /// Starts a new conversation session
        /// - Parameters:
        ///   - config: Session configuration
        ///   - callbacks: Callbacks for conversation events
        ///   - clientTools: Client tools callbacks (optional)
        /// - Returns: A started `Conversation` instance
        public static func startSession(config: SessionConfig, callbacks: Callbacks = Callbacks(), clientTools: ClientTools? = nil) async throws -> Conversation {
            // Step 1: Create the WebSocket connection
            let connection = try await Connection.create(config: config)

            // Step 2: Initialize the Conversation
            let conversation = Conversation(connection: connection, callbacks: callbacks, clientTools: clientTools)

            return conversation
        }

        private func setupWebSocket() {
            callbacks.onConnect(connection.conversationId)
            updateStatus(.connected)

            // Configure WebSocket for larger messages if possible
            if let urlSessionTask = connection.socket as? URLSessionWebSocketTask {
                if #available(iOS 15.0, macOS 12.0, *) {
                    urlSessionTask.maximumMessageSize = Constants.maxRequestedMessageSize
                }
            }

            receiveMessages()
        }

        private func receiveMessages() {
            connection.socket.receive { [weak self] result in
                guard let self = self else { return }

                switch result {
                case let .success(message):

                    self.handleWebSocketMessage(message)
                case let .failure(error):
                    self.logger.error("WebSocket error: \(error.localizedDescription)")
                    self.callbacks.onError("WebSocket error", error)
                    self.endSession()
                }

                if self.status == .connected {
                    self.receiveMessages()
                }
            }
        }

        private func handleWebSocketMessage(_ message: URLSessionWebSocketTask.Message) {
            switch message {
            case let .string(text):

                guard let data = text.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                      let type = json["type"] as? String
                else {
                    callbacks.onError("Invalid message format", nil)
                    return
                }

                switch type {
                case "client_tool_call":
                    handleClientToolCall(json)

                case "interruption":
                    handleInterruptionEvent(json)

                case "agent_response":
                    handleAgentResponseEvent(json)

                case "user_transcript":
                    handleUserTranscriptEvent(json)

                case "audio":
                    handleAudioEvent(json)

                case "ping":
                    handlePingEvent(json)

                case "internal_tentative_agent_response":
                    break

                case "internal_vad_score":
                    break

                case "internal_turn_probability":
                    break

                default:
                    callbacks.onError("Unknown message type", json)
                }

            case .data:
                callbacks.onError("Received unexpected binary message", nil)

            @unknown default:
                callbacks.onError("Received unknown message type", nil)
            }
        }

        private func handleClientToolCall(_ json: [String: Any]) {
            guard let toolCall = json["client_tool_call"] as? [String: Any],
                  let toolName = toolCall["tool_name"] as? String,
                  let toolCallId = toolCall["tool_call_id"] as? String,
                  let parameters = toolCall["parameters"] as? [String: Any]
            else {
                callbacks.onError("Invalid client tool call format", json)
                return
            }

            // Serialize parameters to JSON Data for thread-safety
            let serializedParameters: Data
            do {
                serializedParameters = try JSONSerialization.data(withJSONObject: parameters, options: [])
            } catch {
                callbacks.onError("Failed to serialize parameters", error)
                return
            }

            // Execute in a Task (now safe because of serializedParameters)
            Task { [toolName, toolCallId, serializedParameters] in
                do {
                    // Deserialize within the Task to pass into clientTools.handle
                    let deserializedParameters = try JSONSerialization.jsonObject(with: serializedParameters) as? [String: Any] ?? [:]

                    let result = try await clientTools?.handle(toolName, parameters: deserializedParameters)

                    let response: [String: Any] = [
                        "type": "client_tool_result",
                        "tool_call_id": toolCallId,
                        "result": result ?? "",
                        "is_error": false,
                    ]
                    sendWebSocketMessage(response)
                } catch {
                    let response: [String: Any] = [
                        "type": "client_tool_result",
                        "tool_call_id": toolCallId,
                        "result": error.localizedDescription,
                        "is_error": true,
                    ]
                    sendWebSocketMessage(response)
                }
            }
        }

        private func handleInterruptionEvent(_ json: [String: Any]) {
            guard let event = json["interruption_event"] as? [String: Any],
                  let eventId = event["event_id"] as? Int else { return }

            lastInterruptTimestamp = eventId
            updateMode(.listening)
            callbacks.onInterruption()
        }

        private func handleAgentResponseEvent(_ json: [String: Any]) {
            guard let event = json["agent_response_event"] as? [String: Any],
                  let response = event["agent_response"] as? String else { return }
            callbacks.onMessage(response, .ai)
        }

        private func handleUserTranscriptEvent(_ json: [String: Any]) {
            guard let event = json["user_transcription_event"] as? [String: Any],
                  let transcript = event["user_transcript"] as? String else { return }
            callbacks.onMessage(transcript, .user)
        }

        private func handleAudioEvent(_ json: [String: Any]) {
            guard let event = json["audio_event"] as? [String: Any],
                  let audioBase64 = event["audio_base_64"] as? String,
                  let eventId = event["event_id"] as? Int,
                  lastInterruptTimestamp <= eventId else { return }

            // Check if we need to split the audio chunk for WebSocket size limits
            if audioBase64.utf8.count > Constants.maxWebSocketMessageSize {
                // Split the base64 string into multiple parts to process separately
                let chunkSize = Constants.safeMessageSize
                var offset = 0

                while offset < audioBase64.count {
                    let endIndex = min(offset + chunkSize, audioBase64.count)
                    let startIndex = audioBase64.index(audioBase64.startIndex, offsetBy: offset)
                    let endStringIndex = audioBase64.index(audioBase64.startIndex, offsetBy: endIndex)
                    let subChunk = String(audioBase64[startIndex ..< endStringIndex])

                    addAudioBase64Chunk(subChunk)
                    offset = endIndex
                }
            } else {
                // Process the whole chunk
                addAudioBase64Chunk(audioBase64)
            }

            updateMode(.speaking)
        }

        private func handlePingEvent(_ json: [String: Any]) {
            guard let event = json["ping_event"] as? [String: Any],
                  let eventId = event["event_id"] as? Int else { return }
            let response: [String: Any] = ["type": "pong", "event_id": eventId]
            sendWebSocketMessage(response)
        }

        private func sendWebSocketMessage(_ message: [String: Any]) {
            guard let data = try? JSONSerialization.data(withJSONObject: message),
                  let string = String(data: data, encoding: .utf8)
            else {
                callbacks.onError("Failed to encode message", message)
                return
            }

            connection.socket.send(.string(string)) { [weak self] error in
                if let error = error {
                    self?.logger.error("Failed to send WebSocket message: \(error.localizedDescription)")
                    self?.callbacks.onError("Failed to send WebSocket message", error)
                }
            }
        }
        
        func processUserAudio(data: Data, frameCount: Int) {
            let totalBytes = data.count
            if totalBytes <= Constants.safeMessageSize {
                let base64String = data.base64EncodedString()
                let message: [String: Any] = ["type": "user_audio_chunk", "user_audio_chunk": base64String]
                self.sendWebSocketMessage(message)
            } else {
                // Split into smaller chunks
                let framesPerChunk = Constants.safeMessageSize / MemoryLayout<Int16>.size
                var frameOffset = 0
                
                while frameOffset < frameCount {
                    let framesInChunk = min(framesPerChunk, frameCount - frameOffset)
                    let bytesInChunk = framesInChunk * MemoryLayout<Int16>.size
                    
                    let chunkData = data.advanced(by: frameCount).prefix(bytesInChunk)
                    let base64String = chunkData.base64EncodedString()
                    
                    let message: [String: Any] = ["type": "user_audio_chunk", "user_audio_chunk": base64String]
                    self.sendWebSocketMessage(message)
                    
                    frameOffset += framesInChunk
                }
            }
        }

        private func addAudioBase64Chunk(_ chunk: String) {
            guard let data = ElevenLabsSDK.base64ToArrayBuffer(chunk) else {
                callbacks.onError("Failed to decode audio chunk", nil)
                return
            }
            callbacks.onAudioEvent(data)
        }

        private func updateMode(_ newMode: Mode) {
            guard mode != newMode else { return }
            mode = newMode
            callbacks.onModeChange(newMode)
        }

        private func updateStatus(_ newStatus: Status) {
            guard status != newStatus else { return }
            status = newStatus
            callbacks.onStatusChange(newStatus)
        }

        /// Send a contextual update event
        public func sendContextualUpdate(_ text: String) {
            let event: [String: Any] = [
                "type": "contextual_update",
                "text": text
            ]
            sendWebSocketMessage(event)
        }

        /// Ends the current conversation session
        public func endSession() {
            guard status == .connected else { return }

            updateStatus(.disconnecting)
            connection.close()
            updateStatus(.disconnected)
        }

        /// Retrieves the conversation ID
        /// - Returns: Conversation identifier
        public func getId() -> String {
            connection.conversationId
        }
    }

    // MARK: - Errors

    /// Defines errors specific to ElevenLabsSDK
    public enum ElevenLabsError: Error, LocalizedError {
        case invalidConfiguration
        case invalidURL
        case invalidInitialMessageFormat
        case unexpectedBinaryMessage
        case unknownMessageType
        case failedToCreateAudioFormat
        case failedToCreateAudioComponent
        case failedToCreateAudioComponentInstance

        public var errorDescription: String? {
            switch self {
            case .invalidConfiguration:
                return "Invalid configuration provided."
            case .invalidURL:
                return "The provided URL is invalid."
            case .failedToCreateAudioFormat:
                return "Failed to create the audio format."
            case .failedToCreateAudioComponent:
                return "Failed to create audio component."
            case .failedToCreateAudioComponentInstance:
                return "Failed to create audio component instance."
            case .invalidInitialMessageFormat:
                return "The initial message format is invalid."
            case .unexpectedBinaryMessage:
                return "Received an unexpected binary message."
            case .unknownMessageType:
                return "Received an unknown message type."
            }
        }
    }
}

extension NSLock {
    /// Executes a closure within a locked context
    /// - Parameter body: Closure to execute
    /// - Returns: Result of the closure
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}

private extension Data {
    /// Initializes `Data` from an array of Int16
    /// - Parameter buffer: Array of Int16 values
    init(buffer: [Int16]) {
        self = buffer.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}

extension Encodable {
    var dictionary: [String: Any]? {
        guard let data = try? JSONEncoder().encode(self) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data, options: .allowFragments)) as? [String: Any]
    }
}
