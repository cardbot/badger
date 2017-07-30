/*
 * MIT License
 *
 * Copyright (c) 2017 sean
 * Modifications copyright (C) 2016 Andrew Kimball.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Functions derived from github.com/sean-public/fast-skiplist/skiplist.go.
// Pre-calculating possibilities gives a measurable speedup in the insertion
// case, since repeatedly generating random numbers takes a non-trivial amount
// of time. This method requires only one random number per insert.

package skl

import (
	"math"
)

const (
	maxHeight         = 64
	pValue    float64 = 1 / math.E
)

var (
	probabilities [maxHeight]float64
)

// Calculates a table of probabilities in advance. Each row in the table
// indicates the possibility that a node in the skiplist has the
// corresponding height. Probabilities are between 0 and 1, with a maximum
// allowable height of 64. The P value is set to 1/e, which is optimal for
// a general purpose skiplist.
func init() {
	for i := 0; i < maxHeight; i++ {
		probabilities[i] = math.Pow(pValue, float64(i))
	}
}

// RandomHeight maps a random float64 value in [0.0, 1.0) to the tower height
// that a skiplist node should use, where height <= maxHeight and maxHeight
// is <= 64.
func randomHeight(rnd float64, maxHeight int) int {
	height := 1
	for height < maxHeight && rnd < probabilities[height] {
		height++
	}
	return height
}
