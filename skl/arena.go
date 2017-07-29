/*
 * Copyright 2017 Dgraph Labs, Inc. and Contributors
 * Modifications copyright (C) 2016 Andrew Kimball.
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
	"errors"
	"sync/atomic"
	"unsafe"
)

// Arena should be lock-free.
type Arena struct {
	size uint32
	buf  []byte
}

const (
	kSizeSize    = uint32(unsafe.Sizeof(uint32(0)))
)

var (
	ErrArenaFull = errors.New("allocation failed because arena is full")
)

// NewArena allocates a new arena of the specified size and returns it.
func NewArena(size uint32) *Arena {
	out := &Arena{
		buf: make([]byte, size),
	}

	return out
}

func (a *Arena) Size() uint32 {
	return atomic.LoadUint32(&a.size)
}

func (a *Arena) Reset() {
	atomic.StoreUint32(&a.size, 0)
}

func (a *Arena) Alloc(size uint32) (uint32, error) {
	// The actual size of the allocation includes prepended size bytes.
	actual := size + kSizeSize

	newSize := atomic.AddUint32(&a.size, actual)
	if int(newSize) > len(a.buf) {
		return 0, ErrArenaFull
	}

	offset := newSize - size

	// Write the size bytes just before the offset.
	*(*uint32)(unsafe.Pointer(&a.buf[offset-kSizeSize])) = size

	// Return offset to value portion of the allocation.
	return offset, nil
}

func (a *Arena) GetBytes(offset uint32) []byte {
	if offset == 0 {
		return nil
	}

	size := *(*uint32)(unsafe.Pointer(&a.buf[offset-kSizeSize]))
	return a.buf[offset : offset+size]
}

func (a *Arena) GetSizeBytes(offset, size uint32) []byte {
	if offset == 0 {
		return nil
	}

	return a.buf[offset : offset+size]
}

func (a *Arena) GetPointer(offset uint32) unsafe.Pointer {
	if offset == 0 {
		return nil
	}

	return unsafe.Pointer(&a.buf[offset])
}

func (a *Arena) GetOffsetOf(ptr unsafe.Pointer) uint32 {
	if ptr == nil {
		return 0
	}

	return uint32(uintptr(ptr) - uintptr(unsafe.Pointer(&a.buf[0])))
}
