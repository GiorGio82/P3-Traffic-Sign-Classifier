??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??

~
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer1/kernel
w
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*&
_output_shapes
: *
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
: *
dtype0
~
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_namelayer2/kernel
w
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*&
_output_shapes
: @*
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:@*
dtype0

layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*
shared_namelayer3/kernel
x
!layer3/kernel/Read/ReadVariableOpReadVariableOplayer3/kernel*'
_output_shapes
:@?*
dtype0
o
layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer3/bias
h
layer3/bias/Read/ReadVariableOpReadVariableOplayer3/bias*
_output_shapes	
:?*
dtype0
w
layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	x*
shared_namelayer4/kernel
p
!layer4/kernel/Read/ReadVariableOpReadVariableOplayer4/kernel*
_output_shapes
:	?	x*
dtype0
n
layer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_namelayer4/bias
g
layer4/bias/Read/ReadVariableOpReadVariableOplayer4/bias*
_output_shapes
:x*
dtype0
v
layer5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*
shared_namelayer5/kernel
o
!layer5/kernel/Read/ReadVariableOpReadVariableOplayer5/kernel*
_output_shapes

:xT*
dtype0
n
layer5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*
shared_namelayer5/bias
g
layer5/bias/Read/ReadVariableOpReadVariableOplayer5/bias*
_output_shapes
:T*
dtype0
~
layer5-out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T+*"
shared_namelayer5-out/kernel
w
%layer5-out/kernel/Read/ReadVariableOpReadVariableOplayer5-out/kernel*
_output_shapes

:T+*
dtype0
v
layer5-out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+* 
shared_namelayer5-out/bias
o
#layer5-out/bias/Read/ReadVariableOpReadVariableOplayer5-out/bias*
_output_shapes
:+*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/layer1/kernel/m
?
(Adam/layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/layer1/bias/m
u
&Adam/layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/m*
_output_shapes
: *
dtype0
?
Adam/layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*%
shared_nameAdam/layer2/kernel/m
?
(Adam/layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/m*&
_output_shapes
: @*
dtype0
|
Adam/layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/layer2/bias/m
u
&Adam/layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/layer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*%
shared_nameAdam/layer3/kernel/m
?
(Adam/layer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/m*'
_output_shapes
:@?*
dtype0
}
Adam/layer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/layer3/bias/m
v
&Adam/layer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/layer4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	x*%
shared_nameAdam/layer4/kernel/m
~
(Adam/layer4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer4/kernel/m*
_output_shapes
:	?	x*
dtype0
|
Adam/layer4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*#
shared_nameAdam/layer4/bias/m
u
&Adam/layer4/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer4/bias/m*
_output_shapes
:x*
dtype0
?
Adam/layer5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*%
shared_nameAdam/layer5/kernel/m
}
(Adam/layer5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer5/kernel/m*
_output_shapes

:xT*
dtype0
|
Adam/layer5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*#
shared_nameAdam/layer5/bias/m
u
&Adam/layer5/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer5/bias/m*
_output_shapes
:T*
dtype0
?
Adam/layer5-out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T+*)
shared_nameAdam/layer5-out/kernel/m
?
,Adam/layer5-out/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer5-out/kernel/m*
_output_shapes

:T+*
dtype0
?
Adam/layer5-out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*'
shared_nameAdam/layer5-out/bias/m
}
*Adam/layer5-out/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer5-out/bias/m*
_output_shapes
:+*
dtype0
?
Adam/layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/layer1/kernel/v
?
(Adam/layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/layer1/bias/v
u
&Adam/layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/v*
_output_shapes
: *
dtype0
?
Adam/layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*%
shared_nameAdam/layer2/kernel/v
?
(Adam/layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/v*&
_output_shapes
: @*
dtype0
|
Adam/layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/layer2/bias/v
u
&Adam/layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/layer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*%
shared_nameAdam/layer3/kernel/v
?
(Adam/layer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/v*'
_output_shapes
:@?*
dtype0
}
Adam/layer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/layer3/bias/v
v
&Adam/layer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/layer4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	x*%
shared_nameAdam/layer4/kernel/v
~
(Adam/layer4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer4/kernel/v*
_output_shapes
:	?	x*
dtype0
|
Adam/layer4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*#
shared_nameAdam/layer4/bias/v
u
&Adam/layer4/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer4/bias/v*
_output_shapes
:x*
dtype0
?
Adam/layer5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*%
shared_nameAdam/layer5/kernel/v
}
(Adam/layer5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer5/kernel/v*
_output_shapes

:xT*
dtype0
|
Adam/layer5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*#
shared_nameAdam/layer5/bias/v
u
&Adam/layer5/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer5/bias/v*
_output_shapes
:T*
dtype0
?
Adam/layer5-out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T+*)
shared_nameAdam/layer5-out/kernel/v
?
,Adam/layer5-out/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer5-out/kernel/v*
_output_shapes

:T+*
dtype0
?
Adam/layer5-out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*'
shared_nameAdam/layer5-out/bias/v
}
*Adam/layer5-out/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer5-out/bias/v*
_output_shapes
:+*
dtype0

NoOpNoOp
?M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?M
value?LB?L B?L
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
R
#trainable_variables
$regularization_losses
%	variables
&	keras_api
h

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
R
-trainable_variables
.regularization_losses
/	variables
0	keras_api
R
1trainable_variables
2regularization_losses
3	variables
4	keras_api
h

5kernel
6bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
R
;trainable_variables
<regularization_losses
=	variables
>	keras_api
h

?kernel
@bias
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
R
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
h

Ikernel
Jbias
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
?
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratem?m?m?m?'m?(m?5m?6m??m?@m?Im?Jm?v?v?v?v?'v?(v?5v?6v??v?@v?Iv?Jv?
V
0
1
2
3
'4
(5
56
67
?8
@9
I10
J11
 
V
0
1
2
3
'4
(5
56
67
?8
@9
I10
J11
?
Tnon_trainable_variables
trainable_variables
regularization_losses
Ulayer_regularization_losses
	variables

Vlayers
Wmetrics
Xlayer_metrics
 
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Ynon_trainable_variables
trainable_variables
regularization_losses
Zlayer_regularization_losses
	variables

[layers
\metrics
]layer_metrics
 
 
 
?
^non_trainable_variables
trainable_variables
regularization_losses
_layer_regularization_losses
	variables

`layers
ametrics
blayer_metrics
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
cnon_trainable_variables
trainable_variables
 regularization_losses
dlayer_regularization_losses
!	variables

elayers
fmetrics
glayer_metrics
 
 
 
?
hnon_trainable_variables
#trainable_variables
$regularization_losses
ilayer_regularization_losses
%	variables

jlayers
kmetrics
llayer_metrics
YW
VARIABLE_VALUElayer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
?
mnon_trainable_variables
)trainable_variables
*regularization_losses
nlayer_regularization_losses
+	variables

olayers
pmetrics
qlayer_metrics
 
 
 
?
rnon_trainable_variables
-trainable_variables
.regularization_losses
slayer_regularization_losses
/	variables

tlayers
umetrics
vlayer_metrics
 
 
 
?
wnon_trainable_variables
1trainable_variables
2regularization_losses
xlayer_regularization_losses
3	variables

ylayers
zmetrics
{layer_metrics
YW
VARIABLE_VALUElayer4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61
 

50
61
?
|non_trainable_variables
7trainable_variables
8regularization_losses
}layer_regularization_losses
9	variables

~layers
metrics
?layer_metrics
 
 
 
?
?non_trainable_variables
;trainable_variables
<regularization_losses
 ?layer_regularization_losses
=	variables
?layers
?metrics
?layer_metrics
YW
VARIABLE_VALUElayer5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
 

?0
@1
?
?non_trainable_variables
Atrainable_variables
Bregularization_losses
 ?layer_regularization_losses
C	variables
?layers
?metrics
?layer_metrics
 
 
 
?
?non_trainable_variables
Etrainable_variables
Fregularization_losses
 ?layer_regularization_losses
G	variables
?layers
?metrics
?layer_metrics
][
VARIABLE_VALUElayer5-out/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUElayer5-out/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1
 

I0
J1
?
?non_trainable_variables
Ktrainable_variables
Lregularization_losses
 ?layer_regularization_losses
M	variables
?layers
?metrics
?layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
|z
VARIABLE_VALUEAdam/layer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/layer5-out/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer5-out/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/layer5-out/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer5-out/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_layer1_inputPlaceholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_layer1_inputlayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biaslayer5/kernellayer5/biaslayer5-out/kernellayer5-out/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_21757
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer3/kernel/Read/ReadVariableOplayer3/bias/Read/ReadVariableOp!layer4/kernel/Read/ReadVariableOplayer4/bias/Read/ReadVariableOp!layer5/kernel/Read/ReadVariableOplayer5/bias/Read/ReadVariableOp%layer5-out/kernel/Read/ReadVariableOp#layer5-out/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/layer1/kernel/m/Read/ReadVariableOp&Adam/layer1/bias/m/Read/ReadVariableOp(Adam/layer2/kernel/m/Read/ReadVariableOp&Adam/layer2/bias/m/Read/ReadVariableOp(Adam/layer3/kernel/m/Read/ReadVariableOp&Adam/layer3/bias/m/Read/ReadVariableOp(Adam/layer4/kernel/m/Read/ReadVariableOp&Adam/layer4/bias/m/Read/ReadVariableOp(Adam/layer5/kernel/m/Read/ReadVariableOp&Adam/layer5/bias/m/Read/ReadVariableOp,Adam/layer5-out/kernel/m/Read/ReadVariableOp*Adam/layer5-out/bias/m/Read/ReadVariableOp(Adam/layer1/kernel/v/Read/ReadVariableOp&Adam/layer1/bias/v/Read/ReadVariableOp(Adam/layer2/kernel/v/Read/ReadVariableOp&Adam/layer2/bias/v/Read/ReadVariableOp(Adam/layer3/kernel/v/Read/ReadVariableOp&Adam/layer3/bias/v/Read/ReadVariableOp(Adam/layer4/kernel/v/Read/ReadVariableOp&Adam/layer4/bias/v/Read/ReadVariableOp(Adam/layer5/kernel/v/Read/ReadVariableOp&Adam/layer5/bias/v/Read/ReadVariableOp,Adam/layer5-out/kernel/v/Read/ReadVariableOp*Adam/layer5-out/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_22278
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biaslayer5/kernellayer5/biaslayer5-out/kernellayer5-out/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/layer1/kernel/mAdam/layer1/bias/mAdam/layer2/kernel/mAdam/layer2/bias/mAdam/layer3/kernel/mAdam/layer3/bias/mAdam/layer4/kernel/mAdam/layer4/bias/mAdam/layer5/kernel/mAdam/layer5/bias/mAdam/layer5-out/kernel/mAdam/layer5-out/bias/mAdam/layer1/kernel/vAdam/layer1/bias/vAdam/layer2/kernel/vAdam/layer2/bias/vAdam/layer3/kernel/vAdam/layer3/bias/vAdam/layer4/kernel/vAdam/layer4/bias/vAdam/layer5/kernel/vAdam/layer5/bias/vAdam/layer5-out/kernel/vAdam/layer5-out/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_22423??
?	
?
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21935

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_216912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
D
(__inference_maxPool2_layer_call_fn_21283

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool2_layer_call_and_return_conditional_losses_212772
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21718
layer1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_216912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????  
&
_user_specified_namelayer1_input
?
{
&__inference_layer4_layer_call_fn_22026

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_214082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?	
?
E__inference_layer5-out_layer_call_and_return_conditional_losses_22111

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????+2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_21757
layer1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_212592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????  
&
_user_specified_namelayer1_input
?

b
C__inference_dropout4_layer_call_and_return_conditional_losses_21436

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????x2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????x*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????x2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????x2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????x2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?	
?
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21649
layer1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_216222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????  
&
_user_specified_namelayer1_input
?
D
(__inference_maxPool1_layer_call_fn_21271

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool1_layer_call_and_return_conditional_losses_212652
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

b
C__inference_dropout5_layer_call_and_return_conditional_losses_22085

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????T2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????T*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????T2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????T2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????T:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?	
?
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21906

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_216222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
_
C__inference_maxPool2_layer_call_and_return_conditional_losses_21277

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_22006

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_213892
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_dropout5_layer_call_and_return_conditional_losses_22090

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????T2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????T2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????T:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?4
?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21539
layer1_input
layer1_21321
layer1_21323
layer2_21349
layer2_21351
layer3_21377
layer3_21379
layer4_21419
layer4_21421
layer5_21476
layer5_21478
layer5_out_21533
layer5_out_21535
identity?? dropout4/StatefulPartitionedCall? dropout5/StatefulPartitionedCall?layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?layer4/StatefulPartitionedCall?"layer5-out/StatefulPartitionedCall?layer5/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCalllayer1_inputlayer1_21321layer1_21323*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_213102 
layer1/StatefulPartitionedCall?
maxPool1/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool1_layer_call_and_return_conditional_losses_212652
maxPool1/PartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall!maxPool1/PartitionedCall:output:0layer2_21349layer2_21351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_213382 
layer2/StatefulPartitionedCall?
maxPool2/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool2_layer_call_and_return_conditional_losses_212772
maxPool2/PartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall!maxPool2/PartitionedCall:output:0layer3_21377layer3_21379*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_213662 
layer3/StatefulPartitionedCall?
maxPool3/PartitionedCallPartitionedCall'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool3_layer_call_and_return_conditional_losses_212892
maxPool3/PartitionedCall?
flatten/PartitionedCallPartitionedCall!maxPool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_213892
flatten/PartitionedCall?
layer4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0layer4_21419layer4_21421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_214082 
layer4/StatefulPartitionedCall?
 dropout4/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout4_layer_call_and_return_conditional_losses_214362"
 dropout4/StatefulPartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCall)dropout4/StatefulPartitionedCall:output:0layer5_21476layer5_21478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_214652 
layer5/StatefulPartitionedCall?
 dropout5/StatefulPartitionedCallStatefulPartitionedCall'layer5/StatefulPartitionedCall:output:0!^dropout4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout5_layer_call_and_return_conditional_losses_214932"
 dropout5/StatefulPartitionedCall?
"layer5-out/StatefulPartitionedCallStatefulPartitionedCall)dropout5/StatefulPartitionedCall:output:0layer5_out_21533layer5_out_21535*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer5-out_layer_call_and_return_conditional_losses_215222$
"layer5-out/StatefulPartitionedCall?
IdentityIdentity+layer5-out/StatefulPartitionedCall:output:0!^dropout4/StatefulPartitionedCall!^dropout5/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall#^layer5-out/StatefulPartitionedCall^layer5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2D
 dropout4/StatefulPartitionedCall dropout4/StatefulPartitionedCall2D
 dropout5/StatefulPartitionedCall dropout5/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2H
"layer5-out/StatefulPartitionedCall"layer5-out/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????  
&
_user_specified_namelayer1_input
?
{
&__inference_layer3_layer_call_fn_21995

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_213662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
A__inference_layer5_layer_call_and_return_conditional_losses_21465

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????T2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????T2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?	
?
A__inference_layer5_layer_call_and_return_conditional_losses_22064

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????T2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????T2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
_
C__inference_maxPool1_layer_call_and_return_conditional_losses_21265

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
A__inference_layer4_layer_call_and_return_conditional_losses_22017

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?

*__inference_layer5-out_layer_call_fn_22120

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer5-out_layer_call_and_return_conditional_losses_215222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????T::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
a
(__inference_dropout5_layer_call_fn_22095

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout5_layer_call_and_return_conditional_losses_214932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????T2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????T22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?	
?
A__inference_layer4_layer_call_and_return_conditional_losses_21408

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
a
C__inference_dropout4_layer_call_and_return_conditional_losses_21441

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????x2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????x2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?A
?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21877

inputs)
%layer1_conv2d_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_conv2d_readvariableop_resource*
&layer2_biasadd_readvariableop_resource)
%layer3_conv2d_readvariableop_resource*
&layer3_biasadd_readvariableop_resource)
%layer4_matmul_readvariableop_resource*
&layer4_biasadd_readvariableop_resource)
%layer5_matmul_readvariableop_resource*
&layer5_biasadd_readvariableop_resource-
)layer5_out_matmul_readvariableop_resource.
*layer5_out_biasadd_readvariableop_resource
identity??layer1/BiasAdd/ReadVariableOp?layer1/Conv2D/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/Conv2D/ReadVariableOp?layer3/BiasAdd/ReadVariableOp?layer3/Conv2D/ReadVariableOp?layer4/BiasAdd/ReadVariableOp?layer4/MatMul/ReadVariableOp?!layer5-out/BiasAdd/ReadVariableOp? layer5-out/MatMul/ReadVariableOp?layer5/BiasAdd/ReadVariableOp?layer5/MatMul/ReadVariableOp?
layer1/Conv2D/ReadVariableOpReadVariableOp%layer1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
layer1/Conv2D/ReadVariableOp?
layer1/Conv2DConv2Dinputs$layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
layer1/Conv2D?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
layer1/BiasAdd/ReadVariableOp?
layer1/BiasAddBiasAddlayer1/Conv2D:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
layer1/BiasAddu
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
layer1/Relu?
maxPool1/MaxPoolMaxPoollayer1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
maxPool1/MaxPool?
layer2/Conv2D/ReadVariableOpReadVariableOp%layer2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
layer2/Conv2D/ReadVariableOp?
layer2/Conv2DConv2DmaxPool1/MaxPool:output:0$layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
layer2/Conv2D?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
layer2/BiasAdd/ReadVariableOp?
layer2/BiasAddBiasAddlayer2/Conv2D:output:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
layer2/BiasAddu
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
layer2/Relu?
maxPool2/MaxPoolMaxPoollayer2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
maxPool2/MaxPool?
layer3/Conv2D/ReadVariableOpReadVariableOp%layer3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
layer3/Conv2D/ReadVariableOp?
layer3/Conv2DConv2DmaxPool2/MaxPool:output:0$layer3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
layer3/Conv2D?
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer3/BiasAdd/ReadVariableOp?
layer3/BiasAddBiasAddlayer3/Conv2D:output:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
layer3/BiasAddv
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
layer3/Relu?
maxPool3/MaxPoolMaxPoollayer3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
maxPool3/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapemaxPool3/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
layer4/MatMul/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource*
_output_shapes
:	?	x*
dtype02
layer4/MatMul/ReadVariableOp?
layer4/MatMulMatMulflatten/Reshape:output:0$layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
layer4/MatMul?
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
layer4/BiasAdd/ReadVariableOp?
layer4/BiasAddBiasAddlayer4/MatMul:product:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
layer4/BiasAddm
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
layer4/Relu
dropout4/IdentityIdentitylayer4/Relu:activations:0*
T0*'
_output_shapes
:?????????x2
dropout4/Identity?
layer5/MatMul/ReadVariableOpReadVariableOp%layer5_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
layer5/MatMul/ReadVariableOp?
layer5/MatMulMatMuldropout4/Identity:output:0$layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
layer5/MatMul?
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
layer5/BiasAdd/ReadVariableOp?
layer5/BiasAddBiasAddlayer5/MatMul:product:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
layer5/BiasAddm
layer5/ReluRelulayer5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????T2
layer5/Relu
dropout5/IdentityIdentitylayer5/Relu:activations:0*
T0*'
_output_shapes
:?????????T2
dropout5/Identity?
 layer5-out/MatMul/ReadVariableOpReadVariableOp)layer5_out_matmul_readvariableop_resource*
_output_shapes

:T+*
dtype02"
 layer5-out/MatMul/ReadVariableOp?
layer5-out/MatMulMatMuldropout5/Identity:output:0(layer5-out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
layer5-out/MatMul?
!layer5-out/BiasAdd/ReadVariableOpReadVariableOp*layer5_out_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02#
!layer5-out/BiasAdd/ReadVariableOp?
layer5-out/BiasAddBiasAddlayer5-out/MatMul:product:0)layer5-out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
layer5-out/BiasAdd?
layer5-out/SoftmaxSoftmaxlayer5-out/BiasAdd:output:0*
T0*'
_output_shapes
:?????????+2
layer5-out/Softmax?
IdentityIdentitylayer5-out/Softmax:softmax:0^layer1/BiasAdd/ReadVariableOp^layer1/Conv2D/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/Conv2D/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/Conv2D/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/MatMul/ReadVariableOp"^layer5-out/BiasAdd/ReadVariableOp!^layer5-out/MatMul/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/Conv2D/ReadVariableOplayer1/Conv2D/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/Conv2D/ReadVariableOplayer2/Conv2D/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/Conv2D/ReadVariableOplayer3/Conv2D/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/MatMul/ReadVariableOplayer4/MatMul/ReadVariableOp2F
!layer5-out/BiasAdd/ReadVariableOp!layer5-out/BiasAdd/ReadVariableOp2D
 layer5-out/MatMul/ReadVariableOp layer5-out/MatMul/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/MatMul/ReadVariableOplayer5/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

b
C__inference_dropout5_layer_call_and_return_conditional_losses_21493

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????T2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????T*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????T2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????T2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????T:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_22001

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_dropout4_layer_call_and_return_conditional_losses_22043

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????x2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????x2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?T
?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21824

inputs)
%layer1_conv2d_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_conv2d_readvariableop_resource*
&layer2_biasadd_readvariableop_resource)
%layer3_conv2d_readvariableop_resource*
&layer3_biasadd_readvariableop_resource)
%layer4_matmul_readvariableop_resource*
&layer4_biasadd_readvariableop_resource)
%layer5_matmul_readvariableop_resource*
&layer5_biasadd_readvariableop_resource-
)layer5_out_matmul_readvariableop_resource.
*layer5_out_biasadd_readvariableop_resource
identity??layer1/BiasAdd/ReadVariableOp?layer1/Conv2D/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/Conv2D/ReadVariableOp?layer3/BiasAdd/ReadVariableOp?layer3/Conv2D/ReadVariableOp?layer4/BiasAdd/ReadVariableOp?layer4/MatMul/ReadVariableOp?!layer5-out/BiasAdd/ReadVariableOp? layer5-out/MatMul/ReadVariableOp?layer5/BiasAdd/ReadVariableOp?layer5/MatMul/ReadVariableOp?
layer1/Conv2D/ReadVariableOpReadVariableOp%layer1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
layer1/Conv2D/ReadVariableOp?
layer1/Conv2DConv2Dinputs$layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
layer1/Conv2D?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
layer1/BiasAdd/ReadVariableOp?
layer1/BiasAddBiasAddlayer1/Conv2D:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
layer1/BiasAddu
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
layer1/Relu?
maxPool1/MaxPoolMaxPoollayer1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
maxPool1/MaxPool?
layer2/Conv2D/ReadVariableOpReadVariableOp%layer2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
layer2/Conv2D/ReadVariableOp?
layer2/Conv2DConv2DmaxPool1/MaxPool:output:0$layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
layer2/Conv2D?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
layer2/BiasAdd/ReadVariableOp?
layer2/BiasAddBiasAddlayer2/Conv2D:output:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
layer2/BiasAddu
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
layer2/Relu?
maxPool2/MaxPoolMaxPoollayer2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
maxPool2/MaxPool?
layer3/Conv2D/ReadVariableOpReadVariableOp%layer3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
layer3/Conv2D/ReadVariableOp?
layer3/Conv2DConv2DmaxPool2/MaxPool:output:0$layer3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
layer3/Conv2D?
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
layer3/BiasAdd/ReadVariableOp?
layer3/BiasAddBiasAddlayer3/Conv2D:output:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
layer3/BiasAddv
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
layer3/Relu?
maxPool3/MaxPoolMaxPoollayer3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
maxPool3/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapemaxPool3/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
layer4/MatMul/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource*
_output_shapes
:	?	x*
dtype02
layer4/MatMul/ReadVariableOp?
layer4/MatMulMatMulflatten/Reshape:output:0$layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
layer4/MatMul?
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
layer4/BiasAdd/ReadVariableOp?
layer4/BiasAddBiasAddlayer4/MatMul:product:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
layer4/BiasAddm
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
layer4/Reluu
dropout4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout4/dropout/Const?
dropout4/dropout/MulMullayer4/Relu:activations:0dropout4/dropout/Const:output:0*
T0*'
_output_shapes
:?????????x2
dropout4/dropout/Muly
dropout4/dropout/ShapeShapelayer4/Relu:activations:0*
T0*
_output_shapes
:2
dropout4/dropout/Shape?
-dropout4/dropout/random_uniform/RandomUniformRandomUniformdropout4/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????x*
dtype02/
-dropout4/dropout/random_uniform/RandomUniform?
dropout4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2!
dropout4/dropout/GreaterEqual/y?
dropout4/dropout/GreaterEqualGreaterEqual6dropout4/dropout/random_uniform/RandomUniform:output:0(dropout4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????x2
dropout4/dropout/GreaterEqual?
dropout4/dropout/CastCast!dropout4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????x2
dropout4/dropout/Cast?
dropout4/dropout/Mul_1Muldropout4/dropout/Mul:z:0dropout4/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????x2
dropout4/dropout/Mul_1?
layer5/MatMul/ReadVariableOpReadVariableOp%layer5_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
layer5/MatMul/ReadVariableOp?
layer5/MatMulMatMuldropout4/dropout/Mul_1:z:0$layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
layer5/MatMul?
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
layer5/BiasAdd/ReadVariableOp?
layer5/BiasAddBiasAddlayer5/MatMul:product:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
layer5/BiasAddm
layer5/ReluRelulayer5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????T2
layer5/Reluu
dropout5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout5/dropout/Const?
dropout5/dropout/MulMullayer5/Relu:activations:0dropout5/dropout/Const:output:0*
T0*'
_output_shapes
:?????????T2
dropout5/dropout/Muly
dropout5/dropout/ShapeShapelayer5/Relu:activations:0*
T0*
_output_shapes
:2
dropout5/dropout/Shape?
-dropout5/dropout/random_uniform/RandomUniformRandomUniformdropout5/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????T*
dtype02/
-dropout5/dropout/random_uniform/RandomUniform?
dropout5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
dropout5/dropout/GreaterEqual/y?
dropout5/dropout/GreaterEqualGreaterEqual6dropout5/dropout/random_uniform/RandomUniform:output:0(dropout5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????T2
dropout5/dropout/GreaterEqual?
dropout5/dropout/CastCast!dropout5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????T2
dropout5/dropout/Cast?
dropout5/dropout/Mul_1Muldropout5/dropout/Mul:z:0dropout5/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????T2
dropout5/dropout/Mul_1?
 layer5-out/MatMul/ReadVariableOpReadVariableOp)layer5_out_matmul_readvariableop_resource*
_output_shapes

:T+*
dtype02"
 layer5-out/MatMul/ReadVariableOp?
layer5-out/MatMulMatMuldropout5/dropout/Mul_1:z:0(layer5-out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
layer5-out/MatMul?
!layer5-out/BiasAdd/ReadVariableOpReadVariableOp*layer5_out_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02#
!layer5-out/BiasAdd/ReadVariableOp?
layer5-out/BiasAddBiasAddlayer5-out/MatMul:product:0)layer5-out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
layer5-out/BiasAdd?
layer5-out/SoftmaxSoftmaxlayer5-out/BiasAdd:output:0*
T0*'
_output_shapes
:?????????+2
layer5-out/Softmax?
IdentityIdentitylayer5-out/Softmax:softmax:0^layer1/BiasAdd/ReadVariableOp^layer1/Conv2D/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/Conv2D/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/Conv2D/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/MatMul/ReadVariableOp"^layer5-out/BiasAdd/ReadVariableOp!^layer5-out/MatMul/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/Conv2D/ReadVariableOplayer1/Conv2D/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/Conv2D/ReadVariableOplayer2/Conv2D/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/Conv2D/ReadVariableOplayer3/Conv2D/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/MatMul/ReadVariableOplayer4/MatMul/ReadVariableOp2F
!layer5-out/BiasAdd/ReadVariableOp!layer5-out/BiasAdd/ReadVariableOp2D
 layer5-out/MatMul/ReadVariableOp layer5-out/MatMul/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/MatMul/ReadVariableOplayer5/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
A__inference_layer2_layer_call_and_return_conditional_losses_21338

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
A__inference_layer3_layer_call_and_return_conditional_losses_21986

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
(__inference_dropout4_layer_call_fn_22048

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout4_layer_call_and_return_conditional_losses_214362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????x22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?1
?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21691

inputs
layer1_21654
layer1_21656
layer2_21660
layer2_21662
layer3_21666
layer3_21668
layer4_21673
layer4_21675
layer5_21679
layer5_21681
layer5_out_21685
layer5_out_21687
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?layer4/StatefulPartitionedCall?"layer5-out/StatefulPartitionedCall?layer5/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_21654layer1_21656*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_213102 
layer1/StatefulPartitionedCall?
maxPool1/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool1_layer_call_and_return_conditional_losses_212652
maxPool1/PartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall!maxPool1/PartitionedCall:output:0layer2_21660layer2_21662*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_213382 
layer2/StatefulPartitionedCall?
maxPool2/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool2_layer_call_and_return_conditional_losses_212772
maxPool2/PartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall!maxPool2/PartitionedCall:output:0layer3_21666layer3_21668*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_213662 
layer3/StatefulPartitionedCall?
maxPool3/PartitionedCallPartitionedCall'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool3_layer_call_and_return_conditional_losses_212892
maxPool3/PartitionedCall?
flatten/PartitionedCallPartitionedCall!maxPool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_213892
flatten/PartitionedCall?
layer4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0layer4_21673layer4_21675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_214082 
layer4/StatefulPartitionedCall?
dropout4/PartitionedCallPartitionedCall'layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout4_layer_call_and_return_conditional_losses_214412
dropout4/PartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCall!dropout4/PartitionedCall:output:0layer5_21679layer5_21681*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_214652 
layer5/StatefulPartitionedCall?
dropout5/PartitionedCallPartitionedCall'layer5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout5_layer_call_and_return_conditional_losses_214982
dropout5/PartitionedCall?
"layer5-out/StatefulPartitionedCallStatefulPartitionedCall!dropout5/PartitionedCall:output:0layer5_out_21685layer5_out_21687*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer5-out_layer_call_and_return_conditional_losses_215222$
"layer5-out/StatefulPartitionedCall?
IdentityIdentity+layer5-out/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall#^layer5-out/StatefulPartitionedCall^layer5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2H
"layer5-out/StatefulPartitionedCall"layer5-out/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
A__inference_layer3_layer_call_and_return_conditional_losses_21366

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
{
&__inference_layer1_layer_call_fn_21955

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_213102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
D
(__inference_dropout4_layer_call_fn_22053

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout4_layer_call_and_return_conditional_losses_214412
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?

?
A__inference_layer1_layer_call_and_return_conditional_losses_21310

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
_
C__inference_maxPool3_layer_call_and_return_conditional_losses_21289

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
E__inference_layer5-out_layer_call_and_return_conditional_losses_21522

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????+2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
{
&__inference_layer2_layer_call_fn_21975

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_213382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
A__inference_layer2_layer_call_and_return_conditional_losses_21966

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
a
C__inference_dropout5_layer_call_and_return_conditional_losses_21498

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????T2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????T2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????T:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?

b
C__inference_dropout4_layer_call_and_return_conditional_losses_22038

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????x2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????x*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????x2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????x2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????x2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_21389

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_maxPool3_layer_call_fn_21295

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool3_layer_call_and_return_conditional_losses_212892
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
A__inference_layer1_layer_call_and_return_conditional_losses_21946

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?4
?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21622

inputs
layer1_21585
layer1_21587
layer2_21591
layer2_21593
layer3_21597
layer3_21599
layer4_21604
layer4_21606
layer5_21610
layer5_21612
layer5_out_21616
layer5_out_21618
identity?? dropout4/StatefulPartitionedCall? dropout5/StatefulPartitionedCall?layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?layer4/StatefulPartitionedCall?"layer5-out/StatefulPartitionedCall?layer5/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_21585layer1_21587*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_213102 
layer1/StatefulPartitionedCall?
maxPool1/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool1_layer_call_and_return_conditional_losses_212652
maxPool1/PartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall!maxPool1/PartitionedCall:output:0layer2_21591layer2_21593*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_213382 
layer2/StatefulPartitionedCall?
maxPool2/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool2_layer_call_and_return_conditional_losses_212772
maxPool2/PartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall!maxPool2/PartitionedCall:output:0layer3_21597layer3_21599*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_213662 
layer3/StatefulPartitionedCall?
maxPool3/PartitionedCallPartitionedCall'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool3_layer_call_and_return_conditional_losses_212892
maxPool3/PartitionedCall?
flatten/PartitionedCallPartitionedCall!maxPool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_213892
flatten/PartitionedCall?
layer4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0layer4_21604layer4_21606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_214082 
layer4/StatefulPartitionedCall?
 dropout4/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout4_layer_call_and_return_conditional_losses_214362"
 dropout4/StatefulPartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCall)dropout4/StatefulPartitionedCall:output:0layer5_21610layer5_21612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_214652 
layer5/StatefulPartitionedCall?
 dropout5/StatefulPartitionedCallStatefulPartitionedCall'layer5/StatefulPartitionedCall:output:0!^dropout4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout5_layer_call_and_return_conditional_losses_214932"
 dropout5/StatefulPartitionedCall?
"layer5-out/StatefulPartitionedCallStatefulPartitionedCall)dropout5/StatefulPartitionedCall:output:0layer5_out_21616layer5_out_21618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer5-out_layer_call_and_return_conditional_losses_215222$
"layer5-out/StatefulPartitionedCall?
IdentityIdentity+layer5-out/StatefulPartitionedCall:output:0!^dropout4/StatefulPartitionedCall!^dropout5/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall#^layer5-out/StatefulPartitionedCall^layer5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2D
 dropout4/StatefulPartitionedCall dropout4/StatefulPartitionedCall2D
 dropout5/StatefulPartitionedCall dropout5/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2H
"layer5-out/StatefulPartitionedCall"layer5-out/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
ɼ
?
!__inference__traced_restore_22423
file_prefix"
assignvariableop_layer1_kernel"
assignvariableop_1_layer1_bias$
 assignvariableop_2_layer2_kernel"
assignvariableop_3_layer2_bias$
 assignvariableop_4_layer3_kernel"
assignvariableop_5_layer3_bias$
 assignvariableop_6_layer4_kernel"
assignvariableop_7_layer4_bias$
 assignvariableop_8_layer5_kernel"
assignvariableop_9_layer5_bias)
%assignvariableop_10_layer5_out_kernel'
#assignvariableop_11_layer5_out_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1,
(assignvariableop_21_adam_layer1_kernel_m*
&assignvariableop_22_adam_layer1_bias_m,
(assignvariableop_23_adam_layer2_kernel_m*
&assignvariableop_24_adam_layer2_bias_m,
(assignvariableop_25_adam_layer3_kernel_m*
&assignvariableop_26_adam_layer3_bias_m,
(assignvariableop_27_adam_layer4_kernel_m*
&assignvariableop_28_adam_layer4_bias_m,
(assignvariableop_29_adam_layer5_kernel_m*
&assignvariableop_30_adam_layer5_bias_m0
,assignvariableop_31_adam_layer5_out_kernel_m.
*assignvariableop_32_adam_layer5_out_bias_m,
(assignvariableop_33_adam_layer1_kernel_v*
&assignvariableop_34_adam_layer1_bias_v,
(assignvariableop_35_adam_layer2_kernel_v*
&assignvariableop_36_adam_layer2_bias_v,
(assignvariableop_37_adam_layer3_kernel_v*
&assignvariableop_38_adam_layer3_bias_v,
(assignvariableop_39_adam_layer4_kernel_v*
&assignvariableop_40_adam_layer4_bias_v,
(assignvariableop_41_adam_layer5_kernel_v*
&assignvariableop_42_adam_layer5_bias_v0
,assignvariableop_43_adam_layer5_out_kernel_v.
*assignvariableop_44_adam_layer5_out_bias_v
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_layer5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp%assignvariableop_10_layer5_out_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_layer5_out_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_layer1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_layer1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_layer2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_layer2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_layer3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_layer3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_layer4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_layer4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_layer5_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_layer5_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_layer5_out_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_layer5_out_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_layer1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_layer1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_layer2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_layer2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_layer3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_layer3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_layer4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_layer4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_layer5_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_layer5_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_layer5_out_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_layer5_out_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45?
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
{
&__inference_layer5_layer_call_fn_22073

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_214652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????T2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?x
?
 __inference__wrapped_model_21259
layer1_inputO
Kc132_c264_c3128_k3_p40_3_p50_2_estrue_layer1_conv2d_readvariableop_resourceP
Lc132_c264_c3128_k3_p40_3_p50_2_estrue_layer1_biasadd_readvariableop_resourceO
Kc132_c264_c3128_k3_p40_3_p50_2_estrue_layer2_conv2d_readvariableop_resourceP
Lc132_c264_c3128_k3_p40_3_p50_2_estrue_layer2_biasadd_readvariableop_resourceO
Kc132_c264_c3128_k3_p40_3_p50_2_estrue_layer3_conv2d_readvariableop_resourceP
Lc132_c264_c3128_k3_p40_3_p50_2_estrue_layer3_biasadd_readvariableop_resourceO
Kc132_c264_c3128_k3_p40_3_p50_2_estrue_layer4_matmul_readvariableop_resourceP
Lc132_c264_c3128_k3_p40_3_p50_2_estrue_layer4_biasadd_readvariableop_resourceO
Kc132_c264_c3128_k3_p40_3_p50_2_estrue_layer5_matmul_readvariableop_resourceP
Lc132_c264_c3128_k3_p40_3_p50_2_estrue_layer5_biasadd_readvariableop_resourceS
Oc132_c264_c3128_k3_p40_3_p50_2_estrue_layer5_out_matmul_readvariableop_resourceT
Pc132_c264_c3128_k3_p40_3_p50_2_estrue_layer5_out_biasadd_readvariableop_resource
identity??Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/BiasAdd/ReadVariableOp?Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Conv2D/ReadVariableOp?Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/BiasAdd/ReadVariableOp?Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Conv2D/ReadVariableOp?Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/BiasAdd/ReadVariableOp?Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Conv2D/ReadVariableOp?Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/BiasAdd/ReadVariableOp?Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/MatMul/ReadVariableOp?Gc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/BiasAdd/ReadVariableOp?Fc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/MatMul/ReadVariableOp?Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/BiasAdd/ReadVariableOp?Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/MatMul/ReadVariableOp?
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Conv2D/ReadVariableOpReadVariableOpKc132_c264_c3128_k3_p40_3_p50_2_estrue_layer1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02D
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Conv2D/ReadVariableOp?
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Conv2DConv2Dlayer1_inputJc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
25
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Conv2D?
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/BiasAdd/ReadVariableOpReadVariableOpLc132_c264_c3128_k3_p40_3_p50_2_estrue_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/BiasAdd/ReadVariableOp?
4c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/BiasAddBiasAdd<c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Conv2D:output:0Kc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 26
4c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/BiasAdd?
1c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/ReluRelu=c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 23
1c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Relu?
6c132_c264_c3128_k3_p40.3_p50.2_esTrue/maxPool1/MaxPoolMaxPool?c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
28
6c132_c264_c3128_k3_p40.3_p50.2_esTrue/maxPool1/MaxPool?
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Conv2D/ReadVariableOpReadVariableOpKc132_c264_c3128_k3_p40_3_p50_2_estrue_layer2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02D
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Conv2D/ReadVariableOp?
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Conv2DConv2D?c132_c264_c3128_k3_p40.3_p50.2_esTrue/maxPool1/MaxPool:output:0Jc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
25
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Conv2D?
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/BiasAdd/ReadVariableOpReadVariableOpLc132_c264_c3128_k3_p40_3_p50_2_estrue_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02E
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/BiasAdd/ReadVariableOp?
4c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/BiasAddBiasAdd<c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Conv2D:output:0Kc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@26
4c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/BiasAdd?
1c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/ReluRelu=c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@23
1c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Relu?
6c132_c264_c3128_k3_p40.3_p50.2_esTrue/maxPool2/MaxPoolMaxPool?c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
28
6c132_c264_c3128_k3_p40.3_p50.2_esTrue/maxPool2/MaxPool?
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Conv2D/ReadVariableOpReadVariableOpKc132_c264_c3128_k3_p40_3_p50_2_estrue_layer3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02D
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Conv2D/ReadVariableOp?
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Conv2DConv2D?c132_c264_c3128_k3_p40.3_p50.2_esTrue/maxPool2/MaxPool:output:0Jc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
25
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Conv2D?
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/BiasAdd/ReadVariableOpReadVariableOpLc132_c264_c3128_k3_p40_3_p50_2_estrue_layer3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/BiasAdd/ReadVariableOp?
4c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/BiasAddBiasAdd<c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Conv2D:output:0Kc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????26
4c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/BiasAdd?
1c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/ReluRelu=c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????23
1c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Relu?
6c132_c264_c3128_k3_p40.3_p50.2_esTrue/maxPool3/MaxPoolMaxPool?c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
28
6c132_c264_c3128_k3_p40.3_p50.2_esTrue/maxPool3/MaxPool?
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  25
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/flatten/Const?
5c132_c264_c3128_k3_p40.3_p50.2_esTrue/flatten/ReshapeReshape?c132_c264_c3128_k3_p40.3_p50.2_esTrue/maxPool3/MaxPool:output:0<c132_c264_c3128_k3_p40.3_p50.2_esTrue/flatten/Const:output:0*
T0*(
_output_shapes
:??????????	27
5c132_c264_c3128_k3_p40.3_p50.2_esTrue/flatten/Reshape?
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/MatMul/ReadVariableOpReadVariableOpKc132_c264_c3128_k3_p40_3_p50_2_estrue_layer4_matmul_readvariableop_resource*
_output_shapes
:	?	x*
dtype02D
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/MatMul/ReadVariableOp?
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/MatMulMatMul>c132_c264_c3128_k3_p40.3_p50.2_esTrue/flatten/Reshape:output:0Jc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x25
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/MatMul?
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/BiasAdd/ReadVariableOpReadVariableOpLc132_c264_c3128_k3_p40_3_p50_2_estrue_layer4_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02E
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/BiasAdd/ReadVariableOp?
4c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/BiasAddBiasAdd=c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/MatMul:product:0Kc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x26
4c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/BiasAdd?
1c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/ReluRelu=c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x23
1c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/Relu?
7c132_c264_c3128_k3_p40.3_p50.2_esTrue/dropout4/IdentityIdentity?c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/Relu:activations:0*
T0*'
_output_shapes
:?????????x29
7c132_c264_c3128_k3_p40.3_p50.2_esTrue/dropout4/Identity?
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/MatMul/ReadVariableOpReadVariableOpKc132_c264_c3128_k3_p40_3_p50_2_estrue_layer5_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02D
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/MatMul/ReadVariableOp?
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/MatMulMatMul@c132_c264_c3128_k3_p40.3_p50.2_esTrue/dropout4/Identity:output:0Jc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T25
3c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/MatMul?
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/BiasAdd/ReadVariableOpReadVariableOpLc132_c264_c3128_k3_p40_3_p50_2_estrue_layer5_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02E
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/BiasAdd/ReadVariableOp?
4c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/BiasAddBiasAdd=c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/MatMul:product:0Kc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T26
4c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/BiasAdd?
1c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/ReluRelu=c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????T23
1c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/Relu?
7c132_c264_c3128_k3_p40.3_p50.2_esTrue/dropout5/IdentityIdentity?c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/Relu:activations:0*
T0*'
_output_shapes
:?????????T29
7c132_c264_c3128_k3_p40.3_p50.2_esTrue/dropout5/Identity?
Fc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/MatMul/ReadVariableOpReadVariableOpOc132_c264_c3128_k3_p40_3_p50_2_estrue_layer5_out_matmul_readvariableop_resource*
_output_shapes

:T+*
dtype02H
Fc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/MatMul/ReadVariableOp?
7c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/MatMulMatMul@c132_c264_c3128_k3_p40.3_p50.2_esTrue/dropout5/Identity:output:0Nc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+29
7c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/MatMul?
Gc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/BiasAdd/ReadVariableOpReadVariableOpPc132_c264_c3128_k3_p40_3_p50_2_estrue_layer5_out_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02I
Gc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/BiasAdd/ReadVariableOp?
8c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/BiasAddBiasAddAc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/MatMul:product:0Oc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????+2:
8c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/BiasAdd?
8c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/SoftmaxSoftmaxAc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/BiasAdd:output:0*
T0*'
_output_shapes
:?????????+2:
8c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/Softmax?
IdentityIdentityBc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/Softmax:softmax:0D^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/BiasAdd/ReadVariableOpC^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Conv2D/ReadVariableOpD^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/BiasAdd/ReadVariableOpC^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Conv2D/ReadVariableOpD^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/BiasAdd/ReadVariableOpC^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Conv2D/ReadVariableOpD^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/BiasAdd/ReadVariableOpC^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/MatMul/ReadVariableOpH^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/BiasAdd/ReadVariableOpG^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/MatMul/ReadVariableOpD^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/BiasAdd/ReadVariableOpC^c132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2?
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/BiasAdd/ReadVariableOpCc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/BiasAdd/ReadVariableOp2?
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Conv2D/ReadVariableOpBc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer1/Conv2D/ReadVariableOp2?
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/BiasAdd/ReadVariableOpCc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/BiasAdd/ReadVariableOp2?
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Conv2D/ReadVariableOpBc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer2/Conv2D/ReadVariableOp2?
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/BiasAdd/ReadVariableOpCc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/BiasAdd/ReadVariableOp2?
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Conv2D/ReadVariableOpBc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer3/Conv2D/ReadVariableOp2?
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/BiasAdd/ReadVariableOpCc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/BiasAdd/ReadVariableOp2?
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/MatMul/ReadVariableOpBc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer4/MatMul/ReadVariableOp2?
Gc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/BiasAdd/ReadVariableOpGc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/BiasAdd/ReadVariableOp2?
Fc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/MatMul/ReadVariableOpFc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5-out/MatMul/ReadVariableOp2?
Cc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/BiasAdd/ReadVariableOpCc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/BiasAdd/ReadVariableOp2?
Bc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/MatMul/ReadVariableOpBc132_c264_c3128_k3_p40.3_p50.2_esTrue/layer5/MatMul/ReadVariableOp:] Y
/
_output_shapes
:?????????  
&
_user_specified_namelayer1_input
?1
?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21579
layer1_input
layer1_21542
layer1_21544
layer2_21548
layer2_21550
layer3_21554
layer3_21556
layer4_21561
layer4_21563
layer5_21567
layer5_21569
layer5_out_21573
layer5_out_21575
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer3/StatefulPartitionedCall?layer4/StatefulPartitionedCall?"layer5-out/StatefulPartitionedCall?layer5/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCalllayer1_inputlayer1_21542layer1_21544*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_213102 
layer1/StatefulPartitionedCall?
maxPool1/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool1_layer_call_and_return_conditional_losses_212652
maxPool1/PartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall!maxPool1/PartitionedCall:output:0layer2_21548layer2_21550*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_213382 
layer2/StatefulPartitionedCall?
maxPool2/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool2_layer_call_and_return_conditional_losses_212772
maxPool2/PartitionedCall?
layer3/StatefulPartitionedCallStatefulPartitionedCall!maxPool2/PartitionedCall:output:0layer3_21554layer3_21556*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_213662 
layer3/StatefulPartitionedCall?
maxPool3/PartitionedCallPartitionedCall'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxPool3_layer_call_and_return_conditional_losses_212892
maxPool3/PartitionedCall?
flatten/PartitionedCallPartitionedCall!maxPool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_213892
flatten/PartitionedCall?
layer4/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0layer4_21561layer4_21563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_214082 
layer4/StatefulPartitionedCall?
dropout4/PartitionedCallPartitionedCall'layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout4_layer_call_and_return_conditional_losses_214412
dropout4/PartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCall!dropout4/PartitionedCall:output:0layer5_21567layer5_21569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_214652 
layer5/StatefulPartitionedCall?
dropout5/PartitionedCallPartitionedCall'layer5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout5_layer_call_and_return_conditional_losses_214982
dropout5/PartitionedCall?
"layer5-out/StatefulPartitionedCallStatefulPartitionedCall!dropout5/PartitionedCall:output:0layer5_out_21573layer5_out_21575*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer5-out_layer_call_and_return_conditional_losses_215222$
"layer5-out/StatefulPartitionedCall?
IdentityIdentity+layer5-out/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall#^layer5-out/StatefulPartitionedCall^layer5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????+2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????  ::::::::::::2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2H
"layer5-out/StatefulPartitionedCall"layer5-out/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????  
&
_user_specified_namelayer1_input
?
D
(__inference_dropout5_layer_call_fn_22100

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout5_layer_call_and_return_conditional_losses_214982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????T2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????T:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?\
?
__inference__traced_save_22278
file_prefix,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer3_kernel_read_readvariableop*
&savev2_layer3_bias_read_readvariableop,
(savev2_layer4_kernel_read_readvariableop*
&savev2_layer4_bias_read_readvariableop,
(savev2_layer5_kernel_read_readvariableop*
&savev2_layer5_bias_read_readvariableop0
,savev2_layer5_out_kernel_read_readvariableop.
*savev2_layer5_out_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_layer1_kernel_m_read_readvariableop1
-savev2_adam_layer1_bias_m_read_readvariableop3
/savev2_adam_layer2_kernel_m_read_readvariableop1
-savev2_adam_layer2_bias_m_read_readvariableop3
/savev2_adam_layer3_kernel_m_read_readvariableop1
-savev2_adam_layer3_bias_m_read_readvariableop3
/savev2_adam_layer4_kernel_m_read_readvariableop1
-savev2_adam_layer4_bias_m_read_readvariableop3
/savev2_adam_layer5_kernel_m_read_readvariableop1
-savev2_adam_layer5_bias_m_read_readvariableop7
3savev2_adam_layer5_out_kernel_m_read_readvariableop5
1savev2_adam_layer5_out_bias_m_read_readvariableop3
/savev2_adam_layer1_kernel_v_read_readvariableop1
-savev2_adam_layer1_bias_v_read_readvariableop3
/savev2_adam_layer2_kernel_v_read_readvariableop1
-savev2_adam_layer2_bias_v_read_readvariableop3
/savev2_adam_layer3_kernel_v_read_readvariableop1
-savev2_adam_layer3_bias_v_read_readvariableop3
/savev2_adam_layer4_kernel_v_read_readvariableop1
-savev2_adam_layer4_bias_v_read_readvariableop3
/savev2_adam_layer5_kernel_v_read_readvariableop1
-savev2_adam_layer5_bias_v_read_readvariableop7
3savev2_adam_layer5_out_kernel_v_read_readvariableop5
1savev2_adam_layer5_out_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop(savev2_layer4_kernel_read_readvariableop&savev2_layer4_bias_read_readvariableop(savev2_layer5_kernel_read_readvariableop&savev2_layer5_bias_read_readvariableop,savev2_layer5_out_kernel_read_readvariableop*savev2_layer5_out_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_layer1_kernel_m_read_readvariableop-savev2_adam_layer1_bias_m_read_readvariableop/savev2_adam_layer2_kernel_m_read_readvariableop-savev2_adam_layer2_bias_m_read_readvariableop/savev2_adam_layer3_kernel_m_read_readvariableop-savev2_adam_layer3_bias_m_read_readvariableop/savev2_adam_layer4_kernel_m_read_readvariableop-savev2_adam_layer4_bias_m_read_readvariableop/savev2_adam_layer5_kernel_m_read_readvariableop-savev2_adam_layer5_bias_m_read_readvariableop3savev2_adam_layer5_out_kernel_m_read_readvariableop1savev2_adam_layer5_out_bias_m_read_readvariableop/savev2_adam_layer1_kernel_v_read_readvariableop-savev2_adam_layer1_bias_v_read_readvariableop/savev2_adam_layer2_kernel_v_read_readvariableop-savev2_adam_layer2_bias_v_read_readvariableop/savev2_adam_layer3_kernel_v_read_readvariableop-savev2_adam_layer3_bias_v_read_readvariableop/savev2_adam_layer4_kernel_v_read_readvariableop-savev2_adam_layer4_bias_v_read_readvariableop/savev2_adam_layer5_kernel_v_read_readvariableop-savev2_adam_layer5_bias_v_read_readvariableop3savev2_adam_layer5_out_kernel_v_read_readvariableop1savev2_adam_layer5_out_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:@?:?:	?	x:x:xT:T:T+:+: : : : : : : : : : : : @:@:@?:?:	?	x:x:xT:T:T+:+: : : @:@:@?:?:	?	x:x:xT:T:T+:+: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:%!

_output_shapes
:	?	x: 

_output_shapes
:x:$	 

_output_shapes

:xT: 


_output_shapes
:T:$ 

_output_shapes

:T+: 

_output_shapes
:+:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:%!

_output_shapes
:	?	x: 

_output_shapes
:x:$ 

_output_shapes

:xT: 

_output_shapes
:T:$  

_output_shapes

:T+: !

_output_shapes
:+:,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: @: %

_output_shapes
:@:-&)
'
_output_shapes
:@?:!'

_output_shapes	
:?:%(!

_output_shapes
:	?	x: )

_output_shapes
:x:$* 

_output_shapes

:xT: +

_output_shapes
:T:$, 

_output_shapes

:T+: -

_output_shapes
:+:.

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
layer1_input=
serving_default_layer1_input:0?????????  >

layer5-out0
StatefulPartitionedCall:0?????????+tensorflow/serving/predict:??
?X
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?T
_tf_keras_sequential?T{"class_name": "Sequential", "name": "c132_c264_c3128_k3_p40.3_p50.2_esTrue", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "c132_c264_c3128_k3_p40.3_p50.2_esTrue", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer1_input"}}, {"class_name": "Conv2D", "config": {"name": "layer1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxPool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxPool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxPool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "layer4", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "layer5", "trainable": true, "dtype": "float32", "units": 84, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "layer5-out", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "c132_c264_c3128_k3_p40.3_p50.2_esTrue", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer1_input"}}, {"class_name": "Conv2D", "config": {"name": "layer1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxPool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxPool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxPool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "layer4", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "layer5", "trainable": true, "dtype": "float32", "units": 84, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "layer5-out", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 1]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxPool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxPool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 32]}}
?
#trainable_variables
$regularization_losses
%	variables
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxPool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxPool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 64]}}
?
-trainable_variables
.regularization_losses
/	variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxPool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxPool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
1trainable_variables
2regularization_losses
3	variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

5kernel
6bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer4", "trainable": true, "dtype": "float32", "units": 120, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
?
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
?

?kernel
@bias
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer5", "trainable": true, "dtype": "float32", "units": 84, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}}
?
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

Ikernel
Jbias
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer5-out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer5-out", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 84}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84]}}
?
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratem?m?m?m?'m?(m?5m?6m??m?@m?Im?Jm?v?v?v?v?'v?(v?5v?6v??v?@v?Iv?Jv?"
	optimizer
v
0
1
2
3
'4
(5
56
67
?8
@9
I10
J11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
'4
(5
56
67
?8
@9
I10
J11"
trackable_list_wrapper
?
Tnon_trainable_variables
trainable_variables
regularization_losses
Ulayer_regularization_losses
	variables

Vlayers
Wmetrics
Xlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':% 2layer1/kernel
: 2layer1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Ynon_trainable_variables
trainable_variables
regularization_losses
Zlayer_regularization_losses
	variables

[layers
\metrics
]layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
^non_trainable_variables
trainable_variables
regularization_losses
_layer_regularization_losses
	variables

`layers
ametrics
blayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':% @2layer2/kernel
:@2layer2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
cnon_trainable_variables
trainable_variables
 regularization_losses
dlayer_regularization_losses
!	variables

elayers
fmetrics
glayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
hnon_trainable_variables
#trainable_variables
$regularization_losses
ilayer_regularization_losses
%	variables

jlayers
kmetrics
llayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&@?2layer3/kernel
:?2layer3/bias
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
mnon_trainable_variables
)trainable_variables
*regularization_losses
nlayer_regularization_losses
+	variables

olayers
pmetrics
qlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables
-trainable_variables
.regularization_losses
slayer_regularization_losses
/	variables

tlayers
umetrics
vlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
wnon_trainable_variables
1trainable_variables
2regularization_losses
xlayer_regularization_losses
3	variables

ylayers
zmetrics
{layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?	x2layer4/kernel
:x2layer4/bias
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
|non_trainable_variables
7trainable_variables
8regularization_losses
}layer_regularization_losses
9	variables

~layers
metrics
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
;trainable_variables
<regularization_losses
 ?layer_regularization_losses
=	variables
?layers
?metrics
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:xT2layer5/kernel
:T2layer5/bias
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
?non_trainable_variables
Atrainable_variables
Bregularization_losses
 ?layer_regularization_losses
C	variables
?layers
?metrics
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Etrainable_variables
Fregularization_losses
 ?layer_regularization_losses
G	variables
?layers
?metrics
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!T+2layer5-out/kernel
:+2layer5-out/bias
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
?non_trainable_variables
Ktrainable_variables
Lregularization_losses
 ?layer_regularization_losses
M	variables
?layers
?metrics
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:* 2Adam/layer1/kernel/m
: 2Adam/layer1/bias/m
,:* @2Adam/layer2/kernel/m
:@2Adam/layer2/bias/m
-:+@?2Adam/layer3/kernel/m
:?2Adam/layer3/bias/m
%:#	?	x2Adam/layer4/kernel/m
:x2Adam/layer4/bias/m
$:"xT2Adam/layer5/kernel/m
:T2Adam/layer5/bias/m
(:&T+2Adam/layer5-out/kernel/m
": +2Adam/layer5-out/bias/m
,:* 2Adam/layer1/kernel/v
: 2Adam/layer1/bias/v
,:* @2Adam/layer2/kernel/v
:@2Adam/layer2/bias/v
-:+@?2Adam/layer3/kernel/v
:?2Adam/layer3/bias/v
%:#	?	x2Adam/layer4/kernel/v
:x2Adam/layer4/bias/v
$:"xT2Adam/layer5/kernel/v
:T2Adam/layer5/bias/v
(:&T+2Adam/layer5-out/kernel/v
": +2Adam/layer5-out/bias/v
?2?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21877
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21824
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21539
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21579?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_21259?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+
layer1_input?????????  
?2?
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21649
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21906
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21935
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21718?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_layer1_layer_call_and_return_conditional_losses_21946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer1_layer_call_fn_21955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_maxPool1_layer_call_and_return_conditional_losses_21265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_maxPool1_layer_call_fn_21271?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_layer2_layer_call_and_return_conditional_losses_21966?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer2_layer_call_fn_21975?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_maxPool2_layer_call_and_return_conditional_losses_21277?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_maxPool2_layer_call_fn_21283?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_layer3_layer_call_and_return_conditional_losses_21986?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer3_layer_call_fn_21995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_maxPool3_layer_call_and_return_conditional_losses_21289?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_maxPool3_layer_call_fn_21295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_22001?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_22006?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer4_layer_call_and_return_conditional_losses_22017?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer4_layer_call_fn_22026?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout4_layer_call_and_return_conditional_losses_22043
C__inference_dropout4_layer_call_and_return_conditional_losses_22038?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout4_layer_call_fn_22048
(__inference_dropout4_layer_call_fn_22053?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_layer5_layer_call_and_return_conditional_losses_22064?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer5_layer_call_fn_22073?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout5_layer_call_and_return_conditional_losses_22085
C__inference_dropout5_layer_call_and_return_conditional_losses_22090?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout5_layer_call_fn_22100
(__inference_dropout5_layer_call_fn_22095?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_layer5-out_layer_call_and_return_conditional_losses_22111?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_layer5-out_layer_call_fn_22120?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_21757layer1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_21259?'(56?@IJ=?:
3?0
.?+
layer1_input?????????  
? "7?4
2

layer5-out$?!

layer5-out?????????+?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21539|'(56?@IJE?B
;?8
.?+
layer1_input?????????  
p

 
? "%?"
?
0?????????+
? ?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21579|'(56?@IJE?B
;?8
.?+
layer1_input?????????  
p 

 
? "%?"
?
0?????????+
? ?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21824v'(56?@IJ??<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????+
? ?
`__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_and_return_conditional_losses_21877v'(56?@IJ??<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????+
? ?
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21649o'(56?@IJE?B
;?8
.?+
layer1_input?????????  
p

 
? "??????????+?
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21718o'(56?@IJE?B
;?8
.?+
layer1_input?????????  
p 

 
? "??????????+?
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21906i'(56?@IJ??<
5?2
(?%
inputs?????????  
p

 
? "??????????+?
E__inference_c132_c264_c3128_k3_p40.3_p50.2_esTrue_layer_call_fn_21935i'(56?@IJ??<
5?2
(?%
inputs?????????  
p 

 
? "??????????+?
C__inference_dropout4_layer_call_and_return_conditional_losses_22038\3?0
)?&
 ?
inputs?????????x
p
? "%?"
?
0?????????x
? ?
C__inference_dropout4_layer_call_and_return_conditional_losses_22043\3?0
)?&
 ?
inputs?????????x
p 
? "%?"
?
0?????????x
? {
(__inference_dropout4_layer_call_fn_22048O3?0
)?&
 ?
inputs?????????x
p
? "??????????x{
(__inference_dropout4_layer_call_fn_22053O3?0
)?&
 ?
inputs?????????x
p 
? "??????????x?
C__inference_dropout5_layer_call_and_return_conditional_losses_22085\3?0
)?&
 ?
inputs?????????T
p
? "%?"
?
0?????????T
? ?
C__inference_dropout5_layer_call_and_return_conditional_losses_22090\3?0
)?&
 ?
inputs?????????T
p 
? "%?"
?
0?????????T
? {
(__inference_dropout5_layer_call_fn_22095O3?0
)?&
 ?
inputs?????????T
p
? "??????????T{
(__inference_dropout5_layer_call_fn_22100O3?0
)?&
 ?
inputs?????????T
p 
? "??????????T?
B__inference_flatten_layer_call_and_return_conditional_losses_22001b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????	
? ?
'__inference_flatten_layer_call_fn_22006U8?5
.?+
)?&
inputs??????????
? "???????????	?
A__inference_layer1_layer_call_and_return_conditional_losses_21946l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0????????? 
? ?
&__inference_layer1_layer_call_fn_21955_7?4
-?*
(?%
inputs?????????  
? " ?????????? ?
A__inference_layer2_layer_call_and_return_conditional_losses_21966l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
&__inference_layer2_layer_call_fn_21975_7?4
-?*
(?%
inputs????????? 
? " ??????????@?
A__inference_layer3_layer_call_and_return_conditional_losses_21986m'(7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
&__inference_layer3_layer_call_fn_21995`'(7?4
-?*
(?%
inputs?????????@
? "!????????????
A__inference_layer4_layer_call_and_return_conditional_losses_22017]560?-
&?#
!?
inputs??????????	
? "%?"
?
0?????????x
? z
&__inference_layer4_layer_call_fn_22026P560?-
&?#
!?
inputs??????????	
? "??????????x?
E__inference_layer5-out_layer_call_and_return_conditional_losses_22111\IJ/?,
%?"
 ?
inputs?????????T
? "%?"
?
0?????????+
? }
*__inference_layer5-out_layer_call_fn_22120OIJ/?,
%?"
 ?
inputs?????????T
? "??????????+?
A__inference_layer5_layer_call_and_return_conditional_losses_22064\?@/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????T
? y
&__inference_layer5_layer_call_fn_22073O?@/?,
%?"
 ?
inputs?????????x
? "??????????T?
C__inference_maxPool1_layer_call_and_return_conditional_losses_21265?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_maxPool1_layer_call_fn_21271?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_maxPool2_layer_call_and_return_conditional_losses_21277?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_maxPool2_layer_call_fn_21283?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_maxPool3_layer_call_and_return_conditional_losses_21289?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_maxPool3_layer_call_fn_21295?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
#__inference_signature_wrapper_21757?'(56?@IJM?J
? 
C?@
>
layer1_input.?+
layer1_input?????????  "7?4
2

layer5-out$?!

layer5-out?????????+