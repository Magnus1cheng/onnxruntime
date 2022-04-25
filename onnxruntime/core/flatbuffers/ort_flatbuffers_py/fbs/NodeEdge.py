# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class NodeEdge(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsNodeEdge(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NodeEdge()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def NodeEdgeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # NodeEdge
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # NodeEdge
    def NodeIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # NodeEdge
    def InputEdges(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 12
            from ort_flatbuffers_py.fbs.EdgeEnd import EdgeEnd

            obj = EdgeEnd()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # NodeEdge
    def InputEdgesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # NodeEdge
    def InputEdgesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # NodeEdge
    def OutputEdges(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 12
            from ort_flatbuffers_py.fbs.EdgeEnd import EdgeEnd

            obj = EdgeEnd()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # NodeEdge
    def OutputEdgesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # NodeEdge
    def OutputEdgesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0


def NodeEdgeStart(builder):
    builder.StartObject(3)


def NodeEdgeAddNodeIndex(builder, nodeIndex):
    builder.PrependUint32Slot(0, nodeIndex, 0)


def NodeEdgeAddInputEdges(builder, inputEdges):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(inputEdges), 0)


def NodeEdgeStartInputEdgesVector(builder, numElems):
    return builder.StartVector(12, numElems, 4)


def NodeEdgeAddOutputEdges(builder, outputEdges):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(outputEdges), 0)


def NodeEdgeStartOutputEdgesVector(builder, numElems):
    return builder.StartVector(12, numElems, 4)


def NodeEdgeEnd(builder):
    return builder.EndObject()
