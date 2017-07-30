package skl

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRandomHeight(t *testing.T) {
	require.Equal(t, 64, randomHeight(0.0, 64))
	require.Equal(t, 20, randomHeight(0.0, 20))
	require.Equal(t, 1, randomHeight(0.9, 64))
	require.Equal(t, 1, randomHeight(1/math.E, 64))
	require.Equal(t, 2, randomHeight(0.9999/math.E, 64))
	require.Equal(t, 3, randomHeight(0.9999/math.E/math.E, 64))
	require.Equal(t, 4, randomHeight(0.9999/math.E/math.E/math.E, 64))

	require.Equal(t, 64, randomHeight(-1.0, 64))
	require.Equal(t, 1, randomHeight(2.0, 64))
}
