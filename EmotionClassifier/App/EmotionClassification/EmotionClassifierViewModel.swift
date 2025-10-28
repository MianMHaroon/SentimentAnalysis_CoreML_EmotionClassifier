//
//  EmotionClassifierViewModel.swift
//  EmotionClassifier
//
//  Created by Muhammad Haroon on 28/10/2025.
//

import Foundation

@MainActor
final class EmotionClassifierViewModel: ObservableObject {
    @Published var userInput: String = "" {
        didSet {
            if userInput.isEmpty {
                predictedEmotion = nil
            }
        }
    }
    @Published var predictedEmotion: Emotion?
    @Published var isAnalyzing: Bool = false
    
    func analyzeEmotion() async {
        let trimmed = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        
        isAnalyzing = true
        Task {
            defer { isAnalyzing = false }
            
            do {
                if let result = try await EmotionModelManager.shared.predictEmotion(for: trimmed) {
                    predictedEmotion = Emotion(name: result)
                }
            } catch {
                print("⚠️ Prediction failed: \(error.localizedDescription)")
            }
        }
    }
}

