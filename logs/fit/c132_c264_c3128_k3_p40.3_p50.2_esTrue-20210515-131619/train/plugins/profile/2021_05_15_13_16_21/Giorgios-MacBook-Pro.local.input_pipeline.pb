	?z?G?x@?z?G?x@!?z?G?x@	????"??????"??!????"??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?z?G?x@?????M??A=
ףp?x@Y?E???Ը?*	     ?_@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???x?&??!??`J@)???x?&??1??`J@:Preprocessing2U
Iterator::Model::ParallelMapV2??~j?t??!|?^???-@)??~j?t??1|?^???-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!U*?J?R1@)?? ?rh??1?X,??*@:Preprocessing2F
Iterator::ModelX9??v???!?F??h8@)?~j?t???1?\.???"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!??`0@)y?&1?|?1??`0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!?????~@){?G?zt?1?????~@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????"??IAG?m?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????M???????M??!?????M??      ??!       "      ??!       *      ??!       2	=
ףp?x@=
ףp?x@!=
ףp?x@:      ??!       B      ??!       J	?E???Ը??E???Ը?!?E???Ը?R      ??!       Z	?E???Ը??E???Ը?!?E???Ը?b      ??!       JCPU_ONLYY????"??b qAG?m?X@