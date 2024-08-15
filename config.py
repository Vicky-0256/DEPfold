# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from ast import literal_eval
from configparser import ConfigParser
from typing import Any, Dict, Optional, Sequence

import yaml
from omegaconf import OmegaConf


class Config(object):

    def __init__(self, **kwargs: Any) -> None:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
        # print('__init__')
        super(Config, self).__init__()

        self.update(kwargs)

    def __repr__(self) -> str:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
        # print('__repr__')
        return yaml.dump(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
        # print('__getitem__')
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
        # print('__contains__')
        return hasattr(self, key)

    def __getstate__(self) -> Dict[str, Any]:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
        # print('__getstate__')
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
        # print('__setstate__')
        self.__dict__.update(state)

    @property
    def primitive_config(self) -> Dict[str, Any]:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
        # print('primitive_config')
        from enum import Enum
        from pathlib import Path
        primitive_types = (int, float, bool, str, bytes, Enum, Path)
        return {name: value for name, value in self.__dict__.items() if type(value) in primitive_types}

    def keys(self) -> Any:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
        # print('keys')
        return self.__dict__.keys()

    def items(self) -> Any:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
        # print('items')
        return self.__dict__.items()

    def update(self, kwargs: Dict[str, Any]) -> Config:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
        # print('update')
        for key in ('self', 'cls', '__class__'):
            kwargs.pop(key, None)
        kwargs.update(kwargs.pop('kwargs', dict()))
        for name, value in kwargs.items():
            setattr(self, name, value)
        return self

    # def update(self, kwargs: Dict[str, Any]) -> Config:
    #     # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/config.py')
    #     # print('update')
    #     for key in ('self', 'cls', '__class__'):
    #         kwargs.pop(key, None)
    #     kwargs.update(kwargs.pop('kwargs', dict()))
        
    #     # Set a default value for lr_rate if it is not provided
    #     if 'lr_rate' not in kwargs:
    #         kwargs['lr_rate'] = 0.01  # or whatever default value you want
        
    #     for name, value in kwargs.items():
    #         setattr(self, name, value)
    #     return self

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        return self.__dict__.pop(key, default)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(str(self))

    @classmethod
    def load(cls, conf: str = '', unknown: Optional[Sequence[str]] = None, **kwargs: Any) -> Config:

        if conf and not os.path.exists(conf):
            conf = download(supar.CONFIG['github'].get(conf, conf))
        if conf.endswith(('.yml', '.yaml')):
            config = OmegaConf.load(conf)
        else:
            config = ConfigParser()
            config.read(conf)
            config = dict((name, literal_eval(value)) for s in config.sections() for name, value in config.items(s))
        if unknown is not None:
            parser = argparse.ArgumentParser()
            for name, value in config.items():
                parser.add_argument('--'+name.replace('_', '-'), type=type(value), default=value)
            config.update(vars(parser.parse_args(unknown)))
        return cls(**config).update(kwargs)
