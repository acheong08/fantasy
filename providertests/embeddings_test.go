package providertests

import (
	"context"
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
)

type embeddingProviderBuilderFunc func(t *testing.T, r *vcr.Recorder) (fantasy.Provider, error)

func builderOpenAIEmbeddings(t *testing.T, r *vcr.Recorder) (fantasy.Provider, error) {
	baseURL := "https://api.openai.com/v1"
	if os.Getenv("FANTASY_BASE_URL") != "" {
		baseURL = os.Getenv("FANTASY_BASE_URL")
	}
	apiKey := os.Getenv("FANTASY_OPENAI_API_KEY")
	if os.Getenv("FANTASY_API_KEY") != "" {
		apiKey = os.Getenv("FANTASY_API_KEY")
	}
	provider, err := openai.New(
		openai.WithBaseURL(baseURL),
		openai.WithAPIKey(apiKey),
		openai.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider, nil
}

func embeddingModel() string {
	if os.Getenv("FANTASY_EMBEDDING_MODEL") != "" {
		return os.Getenv("FANTASY_EMBEDDING_MODEL")
	}
	return "text-embedding-3-small"
}

func TestOpenAIEmbeddings(t *testing.T) {
	if os.Getenv("FANTASY_OPENAI_API_KEY") == "" && os.Getenv("FANTASY_API_KEY") == "" {
		t.Skip("FANTASY_API_KEY not set, skipping live test")
	}
	testEmbedding(t, builderOpenAIEmbeddings)
}

func TestOpenAIEmbeddingsWithDimensions(t *testing.T) {
	if os.Getenv("FANTASY_OPENAI_API_KEY") == "" && os.Getenv("FANTASY_API_KEY") == "" {
		t.Skip("FANTASY_API_KEY not set, skipping live test")
	}
	testEmbeddingWithDimensions(t, builderOpenAIEmbeddings)
}

func testEmbedding(t *testing.T, builder embeddingProviderBuilderFunc) {
	t.Run("single input", func(t *testing.T) {
		r := vcr.NewRecorder(t)

		provider, err := builder(t, r)
		require.NoError(t, err, "failed to build provider")

		response, err := provider.Embed(context.Background(), embeddingModel(),
			fantasy.WithEmbeddingInput("The quick brown fox"),
		)
		require.NoError(t, err, "failed to generate embeddings")
		require.NotNil(t, response)
		require.Len(t, response.Embeddings, 1)
		require.NotEmpty(t, response.Embeddings[0].Vector)
		require.Equal(t, embeddingModel(), response.Model)
	})

	t.Run("batch input", func(t *testing.T) {
		r := vcr.NewRecorder(t)

		provider, err := builder(t, r)
		require.NoError(t, err, "failed to build provider")

		response, err := provider.Embed(context.Background(), embeddingModel(),
			fantasy.WithEmbeddingBatch([]string{
				"The quick brown fox",
				"Pack my box with five dozen liquor jugs",
				"How vexingly quick daft zebras jump",
			}),
		)
		require.NoError(t, err, "failed to generate embeddings")
		require.NotNil(t, response)
		require.Len(t, response.Embeddings, 3)
		for i, embedding := range response.Embeddings {
			require.NotEmpty(t, embedding.Vector, "embedding %d should not be empty", i)
			require.Equal(t, i, embedding.Index, "embedding %d should have correct index", i)
		}
		require.Equal(t, embeddingModel(), response.Model)
	})
}

func testEmbeddingWithDimensions(t *testing.T, builder embeddingProviderBuilderFunc) {
	t.Run("with dimensions", func(t *testing.T) {
		r := vcr.NewRecorder(t)

		provider, err := builder(t, r)
		require.NoError(t, err, "failed to build provider")

		response, err := provider.Embed(context.Background(), embeddingModel(),
			fantasy.WithEmbeddingInput("The quick brown fox"),
			fantasy.WithEmbeddingDimensions(256),
		)
		require.NoError(t, err, "failed to generate embeddings")
		require.NotNil(t, response)
		require.Len(t, response.Embeddings, 1)
		require.NotEmpty(t, response.Embeddings[0].Vector)
		require.Equal(t, embeddingModel(), response.Model)
	})
}
