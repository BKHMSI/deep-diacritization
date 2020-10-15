from typing import (
    Tuple,
    List,
    Union,
    Dict,
    Optional,
    Callable,
)
from collections import namedtuple
from abc import ABC, abstractmethod

import torch as T
from torch import nn
from torch.nn import functional as F

from torch import Tensor

import pdb

from dataclasses import dataclass


class IRecurrentCell(ABC, nn.Module):
    @abstractmethod
    def get_init_state(self, input: Tensor):
        pass
    
    @abstractmethod
    def loop(self, inputs, state_t0, mask=None):
        pass

    # def forward(self, input, state, mask=None):
    #     pass

@dataclass
class IRecurrentCellBuilder(ABC):
    hidden_size: int

    def make(self, input_size: int) -> IRecurrentCell:
        pass

    def make_scripted(self, *p, **ks) -> IRecurrentCell:
        return T.jit.script(self.make(*p, **ks))

class RecurrentLayer(nn.Module):
    def reorder_inputs(self, inputs: Union[List[T.Tensor], T.Tensor]):
        #^ inputs : [t b i]
        if self.direction == 'backward':
            return inputs[::-1]
        return inputs

    def __init__(
            self,
            cell: IRecurrentCell,
            direction='forward',
            batch_first=False,
    ):
        super().__init__()
        if isinstance(batch_first, bool):
            batch_first = (batch_first, batch_first)
        self.batch_first = batch_first
        self.direction = direction
        self.cell_: IRecurrentCell = cell

    @T.jit.ignore
    def forward(self, input, state_t0, return_state=None):
        if self.batch_first[0]:
        #^ input : [b t i]
            input = input.transpose(1, 0)
        #^ input : [t b i]
        inputs = input.unbind(0)

        if state_t0 is None:
            state_t0 = self.cell_.get_init_state(input)
    
        inputs = self.reorder_inputs(inputs)

        if return_state:
            sequence, state = self.cell_.loop(inputs, state_t0)
        else:
            sequence, _ = self.cell_.loop(inputs, state_t0)
        #^ sequence : t * [b h]
        sequence = self.reorder_inputs(sequence)
        sequence = T.stack(sequence)
        #^ sequence : [t b h]

        if self.batch_first[1]:
            sequence = sequence.transpose(1, 0)
        #^ sequence : [b t h]  

        if return_state:
            return sequence, state
        else:
            return sequence, None

class BidirectionalRecurrentLayer(nn.Module):
    def __init__(
            self,
            input_size: int,
            cell_builder: IRecurrentCellBuilder,
            batch_first=False,
            return_states=False
    ):
        super().__init__()
        self.batch_first = batch_first
        self.cell_builder = cell_builder
        self.batch_first = batch_first
        self.return_states = return_states
        self.fwd = RecurrentLayer(
            cell_builder.make_scripted(input_size),
            direction='forward',
            batch_first=batch_first
        )
        self.bwd = RecurrentLayer(
            cell_builder.make_scripted(input_size),
            direction='backward',
            batch_first=batch_first
        )

    @T.jit.ignore
    def forward(self, input, state_t0, is_last):
        return_states = is_last and self.return_states
        if return_states:
            fwd, state_fwd = self.fwd(input, state_t0, return_states)
            bwd, state_bwd = self.bwd(input, state_t0, return_states)
            return T.cat([fwd, bwd], dim=-1), (T.cat([state_fwd[0], state_bwd[0]], dim=-1), T.cat([state_fwd[1], state_bwd[1]], dim=-1))
        else:
            fwd, _ = self.fwd(input, state_t0, return_states)
            bwd, _ = self.bwd(input, state_t0, return_states)
            return T.cat([fwd, bwd], dim=-1), None

class RecurrentLayerStack(nn.Module):
    def __init__(
            self,
            cell_builder  : Callable[..., IRecurrentCellBuilder],
            input_size    : int,
            num_layers    : int,
            bidirectional : bool = False,
            batch_first   : bool = False,
            scripted      : bool = True,
            return_states : bool = False,
            *args, **kargs,
    ):
        super().__init__()
        cell_builder_: IRecurrentCellBuilder = cell_builder(*args, **kargs)
        self._cell_builder = cell_builder_

        if bidirectional:
            Dh = cell_builder_.hidden_size * 2
            def make(isize: int, last=False):
                return BidirectionalRecurrentLayer(isize, cell_builder_,
                            batch_first=batch_first, return_states=return_states)
        else:
            Dh = cell_builder_.hidden_size
            def make(isize: int, last=False):
                cell = cell_builder_.make_scripted(isize)
                return RecurrentLayer(cell, isize,
                            batch_first=batch_first)


        if num_layers > 1:
            rnns = [
                make(input_size),
                *[
                    make(Dh)
                    for _ in range(num_layers - 2)
                ],
                make(Dh, last=True)
            ]
        else:
            rnns = [make(input_size, last=True)]

        self.rnn = nn.Sequential(*rnns)

        self.input_size = input_size
        self.hidden_size = self._cell_builder.hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.return_states = return_states

    def __repr__(self):
        return (
            f'${self.__class__.__name__}'
            + '('
            + f'in={self.input_size}, '
            + f'hid={self.hidden_size}, '
            + f'layers={self.num_layers}, '
            + f'bi={self.bidirectional}'
            + '; '
            + str(self._cell_builder)
        )

    def forward(self, input, state_t0=None):
        for layer_idx, rnn in enumerate(self.rnn):
            is_last = (layer_idx == (len(self.rnn) - 1))
            input, state = rnn(input, state_t0, is_last) 
        if self.return_states:
            return input, state  
        return input 
