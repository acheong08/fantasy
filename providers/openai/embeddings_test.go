package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestEmbed_SingleInput(t *testing.T) {
	t.Parallel()

	response := map[string]any{
		"object": "list",
		"data": []map[string]any{
			{
				"object":    "embedding",
				"index":     0,
				"embedding": []float64{0.0023064255, -0.009327292, -0.0028842222},
			},
		},
		"model": "text-embedding-3-small",
		"usage": map[string]int{
			"prompt_tokens": 5,
			"total_tokens":  5,
		},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/embeddings", r.URL.Path)
		require.Equal(t, "POST", r.Method)

		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		require.Equal(t, "text-embedding-3-small", body["model"])
		require.Equal(t, []any{"The quick brown fox"}, body["input"])

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider, err := New(
		WithBaseURL(server.URL),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)

	result, err := provider.Embed(context.Background(), "text-embedding-3-small",
		fantasy.WithEmbeddingInput("The quick brown fox"),
	)
	require.NoError(t, err)
	require.Len(t, result.Embeddings, 1)
	require.Len(t, result.Embeddings[0].Vector, 3)
	require.Equal(t, float32(0.0023064255), result.Embeddings[0].Vector[0])
	require.Equal(t, float32(-0.009327292), result.Embeddings[0].Vector[1])
	require.Equal(t, float32(-0.0028842222), result.Embeddings[0].Vector[2])
	require.Equal(t, 0, result.Embeddings[0].Index)
	require.Equal(t, "text-embedding-3-small", result.Model)
	require.Equal(t, int64(5), result.Usage.InputTokens)
	require.Equal(t, int64(5), result.Usage.TotalTokens)
}

func TestEmbed_BatchInput(t *testing.T) {
	t.Parallel()

	response := map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"object": "embedding", "index": 0, "embedding": []float64{0.1, 0.2}},
			{"object": "embedding", "index": 1, "embedding": []float64{0.3, 0.4}},
			{"object": "embedding", "index": 2, "embedding": []float64{0.5, 0.6}},
		},
		"model": "text-embedding-3-small",
		"usage": map[string]int{
			"prompt_tokens": 15,
			"total_tokens":  15,
		},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/embeddings", r.URL.Path)

		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		input := body["input"].([]any)
		require.Len(t, input, 3)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider, err := New(
		WithBaseURL(server.URL),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)

	result, err := provider.Embed(context.Background(), "text-embedding-3-small",
		fantasy.WithEmbeddingBatch([]string{"text1", "text2", "text3"}),
	)
	require.NoError(t, err)
	require.Len(t, result.Embeddings, 3)

	require.Equal(t, float32(0.1), result.Embeddings[0].Vector[0])
	require.Equal(t, float32(0.2), result.Embeddings[0].Vector[1])
	require.Equal(t, 0, result.Embeddings[0].Index)

	require.Equal(t, float32(0.3), result.Embeddings[1].Vector[0])
	require.Equal(t, float32(0.4), result.Embeddings[1].Vector[1])
	require.Equal(t, 1, result.Embeddings[1].Index)

	require.Equal(t, float32(0.5), result.Embeddings[2].Vector[0])
	require.Equal(t, float32(0.6), result.Embeddings[2].Vector[1])
	require.Equal(t, 2, result.Embeddings[2].Index)
}

func TestEmbed_WithDimensions(t *testing.T) {
	t.Parallel()

	response := map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"object": "embedding", "index": 0, "embedding": []float64{0.1, 0.2}},
		},
		"model": "text-embedding-3-small",
		"usage": map[string]int{
			"prompt_tokens": 5,
			"total_tokens":  5,
		},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/embeddings", r.URL.Path)

		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		require.Equal(t, float64(256), body["dimensions"])

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider, err := New(
		WithBaseURL(server.URL),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)

	result, err := provider.Embed(context.Background(), "text-embedding-3-small",
		fantasy.WithEmbeddingInput("The quick brown fox"),
		fantasy.WithEmbeddingDimensions(256),
	)
	require.NoError(t, err)
	require.Len(t, result.Embeddings, 1)
}

func TestEmbed_EmptyInput(t *testing.T) {
	t.Parallel()

	provider, err := New(
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)

	_, err = provider.Embed(context.Background(), "text-embedding-3-small")
	require.ErrorIs(t, err, fantasy.ErrEmbeddingInputRequired)
}

func TestEmbed_OutputOrder(t *testing.T) {
	t.Parallel()

	response := map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"object": "embedding", "index": 2, "embedding": []float64{0.3}},
			{"object": "embedding", "index": 0, "embedding": []float64{0.1}},
			{"object": "embedding", "index": 1, "embedding": []float64{0.2}},
		},
		"model": "text-embedding-3-small",
		"usage": map[string]int{
			"prompt_tokens": 15,
			"total_tokens":  15,
		},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider, err := New(
		WithBaseURL(server.URL),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)

	result, err := provider.Embed(context.Background(), "text-embedding-3-small",
		fantasy.WithEmbeddingBatch([]string{"a", "b", "c"}),
	)
	require.NoError(t, err)
	require.Len(t, result.Embeddings, 3)

	require.Equal(t, 0, result.Embeddings[0].Index)
	require.Equal(t, float32(0.1), result.Embeddings[0].Vector[0])

	require.Equal(t, 1, result.Embeddings[1].Index)
	require.Equal(t, float32(0.2), result.Embeddings[1].Vector[0])

	require.Equal(t, 2, result.Embeddings[2].Index)
	require.Equal(t, float32(0.3), result.Embeddings[2].Vector[0])
}
