package openai

import (
	"context"
	"slices"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

func (o *provider) Embed(ctx context.Context, modelID string, opts ...fantasy.EmbeddingOption) (*fantasy.EmbeddingResponse, error) {
	call := fantasy.EmbeddingCall{
		Model: modelID,
		Input: []string{},
	}
	for _, opt := range opts {
		opt(&call)
	}

	if len(call.Input) == 0 {
		return nil, fantasy.ErrEmbeddingInputRequired
	}

	openaiClientOptions := make([]option.RequestOption, 0, 5+len(o.options.headers)+len(o.options.sdkOptions))
	openaiClientOptions = append(openaiClientOptions, option.WithMaxRetries(0))

	if o.options.apiKey != "" {
		openaiClientOptions = append(openaiClientOptions, option.WithAPIKey(o.options.apiKey))
	}
	if o.options.baseURL != "" {
		openaiClientOptions = append(openaiClientOptions, option.WithBaseURL(o.options.baseURL))
	}

	for key, value := range o.options.headers {
		openaiClientOptions = append(openaiClientOptions, option.WithHeader(key, value))
	}

	if o.options.client != nil {
		openaiClientOptions = append(openaiClientOptions, option.WithHTTPClient(o.options.client))
	}

	openaiClientOptions = append(openaiClientOptions, o.options.sdkOptions...)

	client := openai.NewClient(openaiClientOptions...)

	params := openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel(modelID),
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: call.Input,
		},
	}

	if call.Dimensions != nil {
		params.Dimensions = openai.Int(int64(*call.Dimensions))
	}

	response, err := client.Embeddings.New(ctx, params)
	if err != nil {
		return nil, err
	}

	embeddings := make([]fantasy.Embedding, len(response.Data))
	for i, e := range response.Data {
		float32Vector := make([]float32, len(e.Embedding))
		for j, f := range e.Embedding {
			float32Vector[j] = float32(f)
		}
		embeddings[i] = fantasy.Embedding{
			Vector: float32Vector,
			Index:  int(e.Index),
		}
	}

	slices.SortFunc(embeddings, func(a, b fantasy.Embedding) int {
		return a.Index - b.Index
	})

	return &fantasy.EmbeddingResponse{
		Embeddings: embeddings,
		Model:      string(response.Model),
		Usage: fantasy.Usage{
			InputTokens: response.Usage.PromptTokens,
			TotalTokens: response.Usage.TotalTokens,
		},
		ProviderMetadata: fantasy.ProviderMetadata{
			Name: &ProviderMetadata{},
		},
	}, nil
}
