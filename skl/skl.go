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

/*
Adapted from RocksDB inline skiplist.

Key differences:
- No optimization for sequential inserts (no "prev").
- No custom comparator.
- Support overwrites. This requires care when we see the same key when inserting.
  For RocksDB or LevelDB, overwrites are implemented as a newer sequence number in the key, so
	there is no need for values. We don't intend to support versioning. In-place updates of values
	would be more efficient.
- We discard all non-concurrent code.
- We do not support Splices. This simplifies the code a lot.
- No AllocateNode or other pointer arithmetic.
- We combine the findLessThan, findGreaterOrEqual, etc into one function.
*/

package skl

import (
	"bytes"
	"errors"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/dgraph-io/badger/y"
)

const (
	kMaxHeight  = 20
	kNodeSize   = uint16(unsafe.Sizeof(node{}))
	kUint32Size = 4
	kDeletedVal = 0
)

var ErrRecordExists = errors.New("record with this key already exists")
var ErrRecordUpdated = errors.New("record was updated by another caller")
var ErrRecordDeleted = errors.New("record was deleted by another caller")

type node struct {
	// A byte slice is 24 bytes. We are trying to save space here.
	keyOffset uint32 // Immutable. No need to lock to access key.
	keySize   uint32

	// Multiple parts of the value are encoded as a single uint64 so that it
	// can be atomically loaded and stored:
	//   uint32: offset
	//   uint16: size
	//   uint16: meta
	value uint64

	// When node is allocated, extra space is allocated after it in memory,
	// and unsafe operations are used to access array elements. This is
	// usually a very small array, since the probability of each successive
	// level decreases exponentially. The size is always <= kMaxHeight. All
	// accesses to elements should use CAS operations, with no need to lock.
	tower [1]uint32
}

type Skiplist struct {
	sync.Mutex

	height uint32 // Current height. 1 <= height <= kMaxHeight. CAS.
	head   *node
	ref    int32
	arena  *Arena
	rng    *rand.Rand
}

func (s *Skiplist) IncrRef() {
	atomic.AddInt32(&s.ref, 1)
}

func (s *Skiplist) DecrRef() {
	newRef := atomic.AddInt32(&s.ref, -1)
	if newRef > 0 {
		return
	}
	s.arena.Reset()
	// Indicate we are closed. Good for testing.  Also, lets GC reclaim memory. Race condition
	// here would suggest we are accessing skiplist when we are supposed to have no reference!
	s.arena = nil
}

func (s *Skiplist) Valid() bool { return s.arena != nil }

func newNode(arena *Arena, key, val []byte, meta uint16, height uint32) (*node, error) {
	keySize := uint16(len(key))
	keyOffset, err := arena.Alloc(keySize, Align1)
	if err != nil {
		return nil, err
	}

	copy(arena.GetBytes(keyOffset, keySize), key)

	// The base level is already allocated in the node struct.
	towerSize := uint16(kUint32Size * (height - 1))
	nodeOffset, err := arena.Alloc(kNodeSize+towerSize, Align8)
	if err != nil {
		return nil, err
	}

	node := (*node)(arena.GetPointer(nodeOffset))
	node.keyOffset = keyOffset
	node.keySize = uint32(keySize)
	node.value, err = node.allocVal(arena, val, meta)
	if err != nil {
		return nil, err
	}

	return node, nil
}

func NewSkiplist(arenaSize uint32) *Skiplist {
	arena := NewArena(arenaSize)
	head, err := newNode(arena, nil, nil, 0, kMaxHeight)
	if err != nil {
		panic("arenaSize is not large enough to hold the head node")
	}

	// Use private random number generator, in order to avoid global lock used
	// by the default rng.
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	skl := &Skiplist{
		height: 1,
		head:   head,
		arena:  arena,
		ref:    1,
		rng:    rng,
	}

	return skl
}

func (s *Skiplist) Height() uint32 {
	return atomic.LoadUint32(&s.height)
}

func (s *Skiplist) Size() uint32 { return s.arena.Size() }

func (s *Skiplist) NewIterator() *Iterator {
	s.IncrRef()
	return &Iterator{list: s, arena: s.arena}
}

func (s *Skiplist) newNode(key, val []byte, meta uint16) (nd *node, height uint32, err error) {
	// We do need to create a new node.
	height = s.randomHeight()
	nd, err = newNode(s.arena, key, val, meta, height)
	if err != nil {
		return
	}

	// Try to increase s.height via CAS.
	listHeight := s.Height()
	for height > listHeight {
		if atomic.CompareAndSwapUint32(&s.height, listHeight, height) {
			// Successfully increased skiplist.height.
			break
		}

		listHeight = s.Height()
	}

	return
}

func (s *Skiplist) getNext(nd *node, h int) *node {
	offset := nd.nextOffset(h)
	return (*node)(s.arena.GetPointer(offset))
}

func (s *Skiplist) randomHeight() uint32 {
	s.Lock()
	defer s.Unlock()

	return uint32(randomHeight(s.rng.Float64(), kMaxHeight))
}

func (n *node) getKey(arena *Arena) []byte {
	return arena.GetBytes(n.keyOffset, uint16(n.keySize))
}

func (n *node) nextOffset(h int) uint32 {
	return atomic.LoadUint32(&n.unsafeTower()[h])
}

func (n *node) casNextOffset(h int, old, val uint32) bool {
	return atomic.CompareAndSwapUint32(&n.unsafeTower()[h], old, val)
}

func (n *node) unsafeTower() *[kMaxHeight]uint32 {
	return (*[kMaxHeight]uint32)(unsafe.Pointer(&n.tower))
}

func (n *node) allocVal(arena *Arena, val []byte, meta uint16) (uint64, error) {
	valSize := uint16(len(val))
	y.AssertTrue(int(valSize) == len(val))

	valOffset, err := arena.Alloc(valSize, Align1)
	if err != nil {
		return 0, err
	}

	copy(arena.GetBytes(valOffset, valSize), val)
	return encodeValue(valOffset, valSize, meta), nil
}

func encodeValue(valOffset uint32, valSize, meta uint16) uint64 {
	return uint64(meta)<<48 | uint64(valSize)<<32 | uint64(valOffset)
}

func decodeValue(value uint64) (valOffset uint32, valSize uint16) {
	valOffset = uint32(value)
	valSize = uint16(value >> 32)
	return
}

func decodeMeta(value uint64) uint16 {
	return uint16(value >> 48)
}

type splice struct {
	prev *node
	next *node
}

func (s *splice) init(prev, next *node) {
	s.prev = prev
	s.next = next
}

// Iterator is an iterator over skiplist object. For new objects, you just
// need to initialize Iterator.list and Iterator.arena.
type Iterator struct {
	list    *Skiplist
	arena   *Arena
	nd      *node
	value   uint64
	fingers [kMaxHeight]uint32
}

func (it *Iterator) Close() error {
	it.list.DecrRef()
	return nil
}

// Valid returns true iff the iterator is positioned at a valid node.
func (it *Iterator) Valid() bool { return it.nd != nil }

// Key returns the key at the current position.
func (it *Iterator) Key() []byte {
	return it.nd.getKey(it.list.arena)
}

// Value returns value.
func (it *Iterator) Value() []byte {
	valOffset, valSize := decodeValue(it.value)
	return it.arena.GetBytes(valOffset, valSize)
}

func (it *Iterator) Meta() uint16 {
	return decodeMeta(it.value)
}

// Next advances to the next position.
func (it *Iterator) Next() {
	next := it.list.getNext(it.nd, 0)
	it.setNode(next)
}

func (it *Iterator) Seek(key []byte) (found bool) {
	level := int(it.list.Height() - 1)

	var prev, next *node
	prev = it.list.head
	for {
		prev, next, found = it.findSpliceForLevel(key, level, prev)
		prevOffset := it.arena.GetPointerOffset(unsafe.Pointer(prev))

		if found {
			for i := level; i >= 0; i-- {
				it.fingers[level] = prevOffset
			}

			break
		}

		it.fingers[level] = prevOffset

		if level == 0 {
			break
		}

		level--
	}

	it.setNode(next)
	return
}

// Add creates a new key/value record if it does not yet exist and positions the
// iterator on it. If the record already exists, then Add positions the iterator
// on the most current value and returns ErrRecordExists. If there isn't
// enough room in the arena, then Add returns ErrArenaFull.
func (it *Iterator) Add(key []byte, val []byte, meta uint16) error {
	var spl [kMaxHeight]splice
	if it.seekForSplice(key, &spl) {
		// Found a matching node, but handle case where it's been deleted.
		return it.setValueIfDeleted(spl[0].next, val, meta)
	}

	nd, height, err := it.list.newNode(key, val, uint16(meta))
	if err != nil {
		return err
	}

	value := nd.value
	ndOffset := it.arena.GetPointerOffset(unsafe.Pointer(nd))

	// We always insert from the base level and up. After you add a node in base
	// level, we cannot create a node in the level above because it would have
	// discovered the node in the base level.
	var found bool
	for i := 0; i < int(height); i++ {
		prev := spl[i].prev
		next := spl[i].next

		if prev == nil {
			// New node increased the height of the skiplist, so assume that the
			// new level has not yet been populated.
			prev = it.list.head
			y.AssertTrue(next == nil)
		}

		for {
			nextOffset := it.arena.GetPointerOffset(unsafe.Pointer(next))
			nd.unsafeTower()[i] = nextOffset

			if prev.casNextOffset(i, nextOffset, ndOffset) {
				// Managed to insert nd between prev and next. Go to the next level.
				break
			}

			// CAS failed. We need to recompute prev and next. It is unlikely to
			// be helpful to try to use a different level as we redo the search,
			// because it is unlikely that lots of nodes are inserted between prev
			// and next.
			prev, next, found = it.findSpliceForLevel(key, i, prev)
			if found {
				y.AssertTruef(i == 0, "Another thread can only race at the base level")
				return it.setValueIfDeleted(next, val, meta)
			}
		}
	}

	it.value = value
	it.nd = nd
	return nil
}

// Set updates the value of the current iteration record if it has not been
// updated or deleted since iterating or seeking to it. If the record has been
// updated, then Set positions the iterator on the most current value and
// returns ErrRecordUpdated. If the record has been deleted, then Set positions
// the iterator on the next non-deleted record and returns ErrRecordDeleted.
func (it *Iterator) Set(val []byte, meta uint16) error {
	new, err := it.nd.allocVal(it.arena, val, meta)
	if err != nil {
		return err
	}

	if !atomic.CompareAndSwapUint64(&it.nd.value, it.value, new) {
		if it.setNode(it.nd) {
			return ErrRecordUpdated
		}

		return ErrRecordDeleted
	}

	it.value = new
	return nil
}

// Delete marks the current iterator record as deleted from the store if it
// has not been updated since iterating or seeking to it. If the record has
// been updated, then Delete positions the iterator on the most current value
// and returns ErrRecordUpdated. If the record is deleted, then Delete positions
// the iterator on the next record.
func (it *Iterator) Delete() error {
	if !atomic.CompareAndSwapUint64(&it.nd.value, it.value, kDeletedVal) {
		if it.setNode(it.nd) {
			return ErrRecordUpdated
		}

		return nil
	}

	// Deletion succeeded, so position iterator on next non-deleted node.
	next := it.list.getNext(it.nd, 0)
	it.setNode(next)
	return nil
}

// SeekToFirst seeks position at the first entry in list.
// Final state of iterator is Valid() iff list is not empty.
func (it *Iterator) SeekToFirst() {
	it.setNode(it.list.getNext(it.list.head, 0))
}

func (it *Iterator) Name() string { return "SkiplistIterator" }

func (it *Iterator) setNode(nd *node) bool {
	var value uint64

	success := true
	for nd != nil {
		// Skip past deleted nodes.
		value = atomic.LoadUint64(&nd.value)
		if value != kDeletedVal {
			break
		}

		success = false
		nd = it.list.getNext(nd, 0)
	}

	it.value = value
	it.nd = nd
	return success
}

func (it *Iterator) setValueIfDeleted(nd *node, val []byte, meta uint16) error {
	var new uint64
	var err error

	for {
		old := atomic.LoadUint64(&nd.value)

		if old != kDeletedVal {
			it.value = old
			it.nd = nd
			return ErrRecordExists
		}

		if new == 0 {
			new, err = nd.allocVal(it.arena, val, meta)
			if err != nil {
				return err
			}
		}

		if atomic.CompareAndSwapUint64(&nd.value, old, new) {
			break
		}
	}

	it.value = new
	it.nd = nd
	return err
}

func (it *Iterator) findSpliceForLevel(key []byte, level int, before *node) (prev, next *node, found bool) {
	prev = before

	for {
		// Assume prev.key < key.
		next = it.list.getNext(prev, level)
		if next == nil {
			break
		}

		nextKey := next.getKey(it.arena)
		cmp := bytes.Compare(key, nextKey)
		if cmp == 0 {
			// Equality case.
			found = true
			break
		}

		if cmp < 0 {
			// We are done for this level, since prevNode.key < key < nextNode.key.
			break
		}

		// Keep moving right on this level.
		prev = next
	}

	return
}

func (it *Iterator) seekForSplice(key []byte, spl *[kMaxHeight]splice) (found bool) {
	var prev, next *node

	level := int(it.list.Height() - 1)
	prev = it.list.head

	useFingers := true
	for {
		if useFingers {
			nd := (*node)(it.arena.GetPointer(it.fingers[level]))
			if nd != nil && bytes.Compare(key, nd.getKey(it.arena)) > 0 {
				prev = nd
			} else {
				useFingers = false
			}
		}

		oldPrev := prev
		prev, next, found = it.findSpliceForLevel(key, level, prev)
		spl[level].init(prev, next)
		it.fingers[level] = it.arena.GetPointerOffset(unsafe.Pointer(prev))

		// If a new value has been stored in the current level's finger, then
		// the iterator has changed position, and so stop using previously stored
		// fingers as a starting position.
		if useFingers && oldPrev != prev {
			useFingers = false
		}

		if level == 0 {
			break
		}

		level--
	}

	return
}

// UniIterator is a unidirectional memtable iterator. It is a thin wrapper around
// Iterator. We like to keep Iterator as before, because it is more powerful and
// we might support bidirectional iterators in the future.
type UniIterator struct {
	iter *Iterator
}

func (s *Skiplist) NewUniIterator(reversed bool) *UniIterator {
	y.AssertTruef(!reversed, "reversed iterator not supported")
	return &UniIterator{iter: s.NewIterator()}
}

func (s *UniIterator) Next() {
	s.iter.Next()
}

func (s *UniIterator) Rewind() {
	s.iter.SeekToFirst()
}

func (s *UniIterator) Seek(key []byte) {
	s.iter.Seek(key)
}

func (s *UniIterator) Value() y.ValueStruct {
	return y.ValueStruct{Value: s.iter.Value(), Meta: byte(s.iter.Meta())}
}

func (s *UniIterator) Key() []byte  { return s.iter.Key() }
func (s *UniIterator) Valid() bool  { return s.iter.Valid() }
func (s *UniIterator) Name() string { return "UniMemtableIterator" }
func (s *UniIterator) Close() error { return s.iter.Close() }
