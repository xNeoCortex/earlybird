/*
 * Copyright 2025 Cosine
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package postprocess

import (
	"math"
	"regexp"
	"strings"
)

var (
	enumeratedLineRegexp = regexp.MustCompile(`(?m)^\s*\d+\s*[\.\)]\s+`)
)

// IsLLMGenerated performs a lightweight heuristic analysis of a file's textual content
// to estimate whether it is likely LLM-generated. It returns a boolean indicating
// suspicion and a confidence level ID (1..5).
//
// Heuristics used:
// - Presence of common LLM disclaimers (e.g., "As an AI language model")
// - Elevated frequency of generic connective phrases (additionally, moreover, etc.)
// - Enumerated "step-by-step" formatting prevalence
// - Low burstiness (low sentence length variance)
//
// This is intentionally conservative and should be tuned over time.
func IsLLMGenerated(raw []byte) (bool, int) {
	if len(raw) < 500 {
		return false, 0
	}

	// Quick printable-text check on a sample window
	sample := raw
	if len(sample) > 8192 {
		sample = raw[:8192]
	}
	printable := 0
	for _, b := range sample {
		if b == 9 || b == 10 || b == 13 || (b >= 32 && b <= 126) {
			printable++
		}
	}
	if float64(printable)/float64(len(sample)) < 0.85 {
		return false, 0
	}

	text := strings.ToLower(string(raw))
	words := splitWords(text)
	totalWords := len(words)
	if totalWords == 0 {
		return false, 0
	}

	// Strong signals
	llmDisclaimers := []string{
		"as an ai language model",
		"as a language model",
		"i am an ai language model",
		"my knowledge cutoff",
		"i cannot browse",
		"i can't browse",
		"i do not have access to",
		"i don't have access to",
		"i cannot provide legal advice",
		"i cannot provide medical advice",
		"here are some steps",
		"step-by-step",
	}
	disclaimerHits := countSubstrings(text, llmDisclaimers)

	// Generic connective/filler phrases commonly overused by LLMs
	fillers := []string{
		"additionally",
		"moreover",
		"furthermore",
		"in conclusion",
		"overall",
		"however",
		"therefore",
		"thus",
		"on the other hand",
		"to summarize",
		"certainly",
		"please let me know",
		"it is important to note",
		"based on the information provided",
		"here are some",
		"by following these steps",
	}
	fillerHits := countSubstrings(text, fillers)
	fillerRatio := float64(fillerHits) / float64(totalWords)

	// Enumerated formatting prevalence
	enumHits := len(enumeratedLineRegexp.FindAllStringIndex(text, -1))

	// Burstiness: sentence length variance vs mean
	sentences := splitSentences(text)
	var cv float64
	if len(sentences) >= 5 {
		lengths := make([]float64, 0, len(sentences))
		for _, s := range sentences {
			lengths = append(lengths, float64(len(splitWords(s))))
		}
		mean, std := meanStd(lengths)
		if mean > 0 {
			cv = std / mean
		}
	}

	// Aggregate a score
	score := 0.0

	if disclaimerHits > 0 {
		score += 0.6
		if disclaimerHits > 1 {
			score += math.Min(0.2, float64(disclaimerHits-1)*0.1)
		}
	}

	// Filler ratio thresholds
	switch {
	case fillerRatio > 0.07:
		score += 0.3
	case fillerRatio > 0.03:
		score += 0.2
	case fillerRatio > 0.01:
		score += 0.1
	}

	// Enumerations
	if enumHits >= 5 {
		score += 0.1
	} else if enumHits >= 3 {
		score += 0.05
	}

	// Burstiness (lower CV => more suspicious)
	if len(sentences) >= 5 {
		if cv < 0.35 {
			score += 0.2
		} else if cv < 0.5 {
			score += 0.1
		}
	}

	if score >= 0.6 {
		// Confidence: stronger when disclaimers are present / very high score
		if disclaimerHits > 0 || score > 0.85 {
			return true, 2 // high
		}
		return true, 3 // medium
	}

	return false, 0
}

func splitWords(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool {
		return r <= ' ' || r == ',' || r == ';' || r == ':' || r == '(' || r == ')' || r == '[' || r == ']' || r == '{' || r == '}' || r == '"' || r == '\'' || r == '|' || r == '/' || r == '\\' || r == '—' || r == '–'
	})
}

func splitSentences(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool {
		return r == '.' || r == '!' || r == '?'
	})
}

func countSubstrings(s string, patterns []string) int {
	count := 0
	for _, p := range patterns {
		// count occurrences of p in s (case-insensitive already)
		idx := 0
		for {
			i := strings.Index(s[idx:], p)
			if i < 0 {
				break
			}
			count++
			idx += i + len(p)
		}
	}
	return count
}

func meanStd(xs []float64) (mean, std float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	mean = sum / float64(len(xs))
	var varSum float64
	for _, x := range xs {
		d := x - mean
		varSum += d * d
	}
	std = math.Sqrt(varSum / float64(len(xs)))
	return mean, std
}