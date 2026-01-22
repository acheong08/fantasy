package fantasy

import (
	"context"
	"errors"
)

var ErrEmbeddingInputRequired = errors.New("embedding input is required")

type EmbeddingOption func(*EmbeddingCall)

type EmbeddingCall struct {
	Model      string
	Input      []string
	Dimensions *int
}

func WithEmbeddingInput(text string) EmbeddingOption {
	return func(c *EmbeddingCall) {
		c.Input = []string{text}
	}
}

func WithEmbeddingBatch(texts []string) EmbeddingOption {
	return func(c *EmbeddingCall) {
		c.Input = texts
	}
}

func WithEmbeddingDimensions(n int) EmbeddingOption {
	return func(c *EmbeddingCall) {
		c.Dimensions = &n
	}
}

type Embedding struct {
	Vector []float32 `json:"vector"`
	Index  int       `json:"index"`
}

type EmbeddingResponse struct {
	Embeddings       []Embedding      `json:"embeddings"`
	Model            string           `json:"model"`
	Usage            Usage            `json:"usage"`
	ProviderMetadata ProviderMetadata `json:"provider_metadata"`
}

type Embedder interface {
	Embed(ctx context.Context, modelID string, opts ...EmbeddingOption) (*EmbeddingResponse, error)
}
