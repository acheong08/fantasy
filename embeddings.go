package fantasy

import (
	"context"
	"errors"
)

// ErrEmbeddingInputRequired is returned when an embedding request has no input.
var ErrEmbeddingInputRequired = errors.New("embedding input is required")

// EmbeddingOption configures an embedding request.
type EmbeddingOption func(*EmbeddingCall)

// EmbeddingCall contains the model input and optional output dimensions for an
// embedding request.
type EmbeddingCall struct {
	Model      string
	Input      []string
	Dimensions *int
}

// WithEmbeddingInput configures a request with one input string.
func WithEmbeddingInput(text string) EmbeddingOption {
	return func(c *EmbeddingCall) {
		c.Input = []string{text}
	}
}

// WithEmbeddingBatch configures a request with a batch of input strings.
func WithEmbeddingBatch(texts []string) EmbeddingOption {
	return func(c *EmbeddingCall) {
		c.Input = texts
	}
}

// WithEmbeddingDimensions requests embeddings with the given number of
// dimensions. Provider and model support varies.
func WithEmbeddingDimensions(n int) EmbeddingOption {
	return func(c *EmbeddingCall) {
		c.Dimensions = &n
	}
}

// Embedding is one vector in an embedding response. Index identifies the
// corresponding position in the request input.
type Embedding struct {
	Vector []float32 `json:"vector"`
	Index  int       `json:"index"`
}

// EmbeddingResponse contains the generated vectors and provider metadata.
type EmbeddingResponse struct {
	Embeddings       []Embedding      `json:"embeddings"`
	Model            string           `json:"model"`
	Usage            Usage            `json:"usage"`
	ProviderMetadata ProviderMetadata `json:"provider_metadata"`
}

// Embedder is implemented by providers that support embedding generation.
// Embedding is an optional provider capability and is intentionally separate
// from Provider, since many language-model providers do not expose embeddings.
type Embedder interface {
	Embed(ctx context.Context, modelID string, opts ...EmbeddingOption) (*EmbeddingResponse, error)
}
