//
//  EmotionClassifierView.swift
//  EmotionClassifier
//
//  Created by Muhammad Haroon on 28/10/2025.
//

import SwiftUI

struct EmotionClassifierView: View {
    @StateObject private var viewModel = EmotionClassifierViewModel()
    @FocusState private var isFocused: Bool
    
    var body: some View {
        ZStack {
            Rectangle()
                .fill(viewModel.predictedEmotion?.backgroundColor ?? Color(.systemBackground))
                .ignoresSafeArea()
                .animation(.easeInOut, value: viewModel.predictedEmotion?.name)
            
            VStack(spacing: 24) {
                Text("🧠 Emotion Classifier")
                    .font(.system(size: 26, weight: .bold))
                    .padding(.top)
                
                TextField("Enter your sentence here...", text: $viewModel.userInput, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .padding()
                    .focused($isFocused)
                    .disabled(viewModel.isAnalyzing)
                    .multilineTextAlignment(.leading)
                    .lineLimit(3...6)
                
                Button(action: {
                    isFocused = false
                    Task { await viewModel.analyzeEmotion() }
                }) {
                    if viewModel.isAnalyzing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                            .frame(maxWidth: .infinity)
                            .padding()
                    } else {
                        Text("Analyze Emotion")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                    }
                }
                .background(Color.accentColor.gradient)
                .foregroundColor(.white)
                .cornerRadius(16)
                .shadow(radius: 4)
                .padding(.horizontal)
                .disabled(viewModel.userInput.trimmingCharacters(in: .whitespaces).isEmpty)
                
                if let emotion = viewModel.predictedEmotion {
                    VStack(spacing: 12) {
                        Text(emotion.emoji)
                            .font(.system(size: 60))
                        Text(emotion.name.capitalized)
                            .font(.title2)
                            .fontWeight(.semibold)
                            .foregroundColor(.primary)
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(.thinMaterial)
                    .cornerRadius(20)
                    .shadow(radius: 8)
                    .transition(.scale.combined(with: .opacity))
                    .animation(.spring(), value: emotion.name)
                }
                
                Spacer()
            }
            .padding()
        }
    }
}

