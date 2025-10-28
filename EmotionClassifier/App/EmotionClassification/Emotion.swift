//
//  Emotion.swift
//  EmotionClassifier
//
//  Created by Muhammad Haroon on 28/10/2025.
//

import SwiftUI

struct Emotion {
    let name: String
    
    var emoji: String {
        switch name.lowercased() {
        case "joy": return "😊"
        case "sadness": return "😢"
        case "anger": return "😡"
        case "fear": return "😨"
        case "love": return "❤️"
        case "surprise": return "😲"
        default: return "🤔"
        }
    }
    
    var color: Color {
        switch name.lowercased() {
        case "joy": return .yellow
        case "sadness": return .blue
        case "anger": return .red
        case "fear": return .purple
        case "love": return .pink
        case "surprise": return .orange
        default: return .gray
        }
    }
    
    var backgroundColor: Color {
        color.opacity(0.25)
    }
}

