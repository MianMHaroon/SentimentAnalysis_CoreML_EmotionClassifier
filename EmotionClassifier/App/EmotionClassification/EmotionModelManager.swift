//
//  EmotionModelManager.swift
//  EmotionClassifier
//
//  Created by Muhammad Haroon on 28/10/2025.
//

import Foundation
import CoreML

actor EmotionModelManager {
    static let shared = EmotionModelManager()
    private var model: EmotionClassifier?
    private let options: NSLinguisticTagger.Options = [.omitWhitespace, .omitPunctuation, .omitOther]
    private lazy var tagger: NSLinguisticTagger = {
        NSLinguisticTagger(
            tagSchemes: NSLinguisticTagger.availableTagSchemes(forLanguage: "en"),
            options: Int(options.rawValue)
        )
    }()
    
    
    private func loadModel() async throws {
        guard model == nil else { return }
        let config = MLModelConfiguration()
        model = try EmotionClassifier(configuration: config)
        print("✅ EmotionClassifier model loaded successfully.")
    }
    
    private func extractFeatures(from text: String) -> [String: Double] {
        var wordCounts = [String: Double]()
        let lowercasedText = text.lowercased()
        tagger.string = lowercasedText
        let range = NSRange(location: 0, length: lowercasedText.utf16.count)
        
        tagger.enumerateTags(in: range, scheme: .nameType, options: options) { _, tokenRange, _, _ in
            let token = (lowercasedText as NSString).substring(with: tokenRange)
            guard token.count >= 2 else { return }
            wordCounts[token, default: 0.0] += 1.0
        }
        
        return wordCounts
    }
    
    func predictEmotion(for text: String) async throws -> String? {
        try await loadModel()
        guard let model = model else {
            print("⚠️ Model not loaded yet.")
            return nil
        }
        
        let features = extractFeatures(from: text)
        guard !features.isEmpty else { return nil }
        
        do {
            let input = EmotionClassifierInput(input: features)
            let output = try model.prediction(input: input)
            
            let emotion = output.classLabel
            
            return (emotion)
        } catch {
            throw error
        }
    }
    
}
