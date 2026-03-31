Signals
=======

Base Classes
------------

.. autoclass:: freqandsee.freqandsee.FlexibleSignal
   :members:
   :special-members: __init__

.. autoclass:: freqandsee.freqandsee.Signal
   :members:
   :special-members: __init__, __add__, __sub__, __mul__, __truediv__, __getitem__

Signal Types
------------

.. autoclass:: freqandsee.freqandsee.DatedSignal
   :members:
   :special-members: __init__

.. autoclass:: freqandsee.freqandsee.ImpulseResponse
   :members:
   :special-members: __init__

.. autoclass:: freqandsee.freqandsee.PSDSignal
   :members:

.. autoclass:: freqandsee.freqandsee.FrequencyResponseSignal
   :members:

Signal Generators
-----------------

.. autoclass:: freqandsee.freqandsee.OneSignal
   :members:
   :special-members: __init__

.. autoclass:: freqandsee.freqandsee.SineSignal
   :members:
   :special-members: __init__

.. autoclass:: freqandsee.freqandsee.WhiteNoise
   :members:
   :special-members: __init__

.. autoclass:: freqandsee.freqandsee.FlickerNoise
   :members:
   :special-members: __init__

Signal Bundle
-------------

.. autoclass:: freqandsee.freqandsee.SignalBundle
   :members:
   :special-members: __init__, __mul__, __matmul__, __getitem__, __iter__, __len__
