/*
 * Copyright 2017 Dgraph Labs, Inc. and Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package skl

import (
	"log"
	"sync/atomic"

	"unsafe"

	"github.com/dgraph-io/badger/y"
	"github.com/pkg/errors"
)

// Arena should be lock-free.
type Arena struct {
	n   uint32
	buf []byte
}

// NewArena returns a new arena.
func NewArena(n int64) *Arena {
	out := &Arena{
		buf: make([]byte, n),
	}
	return out
}

func (s *Arena) Size() int64 {
	return int64(atomic.LoadUint32(&s.n))
}

func (s *Arena) Reset() {
	atomic.StoreUint32(&s.n, 0)
}

func (s *Arena) Alloc(size uint32) uint32 {
	n := atomic.AddUint32(&s.n, size)
	if int(n) > len(s.buf) {
		format := "Arena too small, toWrite:%d newTotal:%d limit:%d"
		log.Fatalf("%+v", errors.Errorf(format, size, n, len(s.buf)))

	}

	return n - size
}

func (s *Arena) GetBytes(offset, size uint32) []byte {
	return s.buf[offset : offset+size]
}

func (s *Arena) GetPointer(offset uint32) unsafe.Pointer {
	return unsafe.Pointer(&s.buf[offset])
}

func (s *Arena) GetOffsetOf(ptr unsafe.Pointer) uint32 {
	return uint32(uintptr(ptr) - uintptr(unsafe.Pointer(&s.buf[0])))
}

// Put will *copy* val into arena. To make better use of this, reuse your input
// val buffer. Returns an offset into buf. User is responsible for remembering
// size of val. We could also store this size inside arena but the encoding and
// decoding will incur some overhead.
func (s *Arena) PutVal(v y.ValueStruct) uint32 {
	l := uint32(v.EncodedSize())
	n := atomic.AddUint32(&s.n, l)
	y.AssertTruef(int(n) <= len(s.buf),
		"Arena too small, toWrite:%d newTotal:%d limit:%d",
		l, n, len(s.buf))
	m := n - l
	v.Encode(s.buf[m:])
	return m
}

func (s *Arena) PutKey(key []byte) uint32 {
	l := uint32(len(key))
	n := atomic.AddUint32(&s.n, l)
	y.AssertTruef(int(n) <= len(s.buf),
		"Arena too small, toWrite:%d newTotal:%d limit:%d",
		l, n, len(s.buf))
	m := n - l
	y.AssertTrue(len(key) == copy(s.buf[m:n], key))
	return m
}

// GetKey returns byte slice at offset.
func (s *Arena) GetKey(offset uint32, size uint16) []byte {
	return s.buf[offset : offset+uint32(size)]
}

// GetVal returns byte slice at offset. The given size should be just the value
// size and should NOT include the meta bytes.
func (s *Arena) GetVal(offset uint32) (ret y.ValueStruct) {
	ret.Decode(s.buf[offset:])
	return
}
