# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class Graph(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsGraph(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Graph()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GraphBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # Graph
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Graph
    def Initializers(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.Tensor import Tensor

            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def InitializersLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def InitializersIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # Graph
    def NodeArgs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.ValueInfo import ValueInfo

            obj = ValueInfo()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def NodeArgsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def NodeArgsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # Graph
    def Nodes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.Node import Node

            obj = Node()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def NodesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def NodesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Graph
    def MaxNodeIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Graph
    def NodeEdges(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.NodeEdge import NodeEdge

            obj = NodeEdge()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def NodeEdgesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def NodeEdgesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # Graph
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Graph
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def InputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # Graph
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Graph
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # Graph
    def SparseInitializers(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.SparseTensor import SparseTensor

            obj = SparseTensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def SparseInitializersLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def SparseInitializersIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

    # Graph
    def RuntimeOptimizations(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.RuntimeOptimizations import RuntimeOptimizations

            obj = RuntimeOptimizations()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None


def GraphStart(builder):
    builder.StartObject(9)


def GraphAddInitializers(builder, initializers):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(initializers), 0)


def GraphStartInitializersVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def GraphAddNodeArgs(builder, nodeArgs):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(nodeArgs), 0)


def GraphStartNodeArgsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def GraphAddNodes(builder, nodes):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(nodes), 0)


def GraphStartNodesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def GraphAddMaxNodeIndex(builder, maxNodeIndex):
    builder.PrependUint32Slot(3, maxNodeIndex, 0)


def GraphAddNodeEdges(builder, nodeEdges):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(nodeEdges), 0)


def GraphStartNodeEdgesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def GraphAddInputs(builder, inputs):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0)


def GraphStartInputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def GraphAddOutputs(builder, outputs):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)


def GraphStartOutputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def GraphAddSparseInitializers(builder, sparseInitializers):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(sparseInitializers), 0)


def GraphStartSparseInitializersVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def GraphAddRuntimeOptimizations(builder, runtimeOptimizations):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(runtimeOptimizations), 0)


def GraphEnd(builder):
    return builder.EndObject()
