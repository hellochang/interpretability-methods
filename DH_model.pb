
;
DeepHit/batch_sizePlaceholder*
dtype0*
shape: 
>
DeepHit/learning_ratePlaceholder*
dtype0*
shape: 
A
DeepHit/keep_probabilityPlaceholder*
dtype0*
shape: 
6
DeepHit/alphaPlaceholder*
dtype0*
shape: 
5
DeepHit/betaPlaceholder*
dtype0*
shape: 
6
DeepHit/gammaPlaceholder*
dtype0*
shape: 
H
DeepHit/inputsPlaceholder*
dtype0*
shape:€€€€€€€€€`
H
DeepHit/labelsPlaceholder*
dtype0*
shape:€€€€€€€€€
N
DeepHit/timetoeventsPlaceholder*
dtype0*
shape:€€€€€€€€€
L
DeepHit/mask1Placeholder*
dtype0*!
shape:€€€€€€€€€П
H
DeepHit/mask2Placeholder*
dtype0*
shape:€€€€€€€€€П
Ђ
BDeepHit/fully_connected/weights/Initializer/truncated_normal/shapeConst*2
_class(
&$loc:@DeepHit/fully_connected/weights*
dtype0*
valueB"`   
   
Ґ
ADeepHit/fully_connected/weights/Initializer/truncated_normal/meanConst*2
_class(
&$loc:@DeepHit/fully_connected/weights*
dtype0*
valueB
 *    
§
CDeepHit/fully_connected/weights/Initializer/truncated_normal/stddevConst*2
_class(
&$loc:@DeepHit/fully_connected/weights*
dtype0*
valueB
 *аз>
Ж
LDeepHit/fully_connected/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalBDeepHit/fully_connected/weights/Initializer/truncated_normal/shape*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights*
dtype0*

seed *
seed2 
Ч
@DeepHit/fully_connected/weights/Initializer/truncated_normal/mulMulLDeepHit/fully_connected/weights/Initializer/truncated_normal/TruncatedNormalCDeepHit/fully_connected/weights/Initializer/truncated_normal/stddev*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights
З
<DeepHit/fully_connected/weights/Initializer/truncated_normalAddV2@DeepHit/fully_connected/weights/Initializer/truncated_normal/mulADeepHit/fully_connected/weights/Initializer/truncated_normal/mean*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights
І
DeepHit/fully_connected/weights
VariableV2*2
_class(
&$loc:@DeepHit/fully_connected/weights*
	container *
dtype0*
shape
:`
*
shared_name 
х
&DeepHit/fully_connected/weights/AssignAssignDeepHit/fully_connected/weights<DeepHit/fully_connected/weights/Initializer/truncated_normal*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights*
use_locking(*
validate_shape(
О
$DeepHit/fully_connected/weights/readIdentityDeepHit/fully_connected/weights*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights
j
1DeepHit/fully_connected/kernel/Regularizer/SquareSquare$DeepHit/fully_connected/weights/read*
T0
e
0DeepHit/fully_connected/kernel/Regularizer/ConstConst*
dtype0*
valueB"       
ј
.DeepHit/fully_connected/kernel/Regularizer/SumSum1DeepHit/fully_connected/kernel/Regularizer/Square0DeepHit/fully_connected/kernel/Regularizer/Const*
T0*

Tidx0*
	keep_dims( 
]
0DeepHit/fully_connected/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *ЈQ8
†
.DeepHit/fully_connected/kernel/Regularizer/mulMul0DeepHit/fully_connected/kernel/Regularizer/mul/x.DeepHit/fully_connected/kernel/Regularizer/Sum*
T0
Ф
0DeepHit/fully_connected/biases/Initializer/zerosConst*1
_class'
%#loc:@DeepHit/fully_connected/biases*
dtype0*
valueB
*    
°
DeepHit/fully_connected/biases
VariableV2*1
_class'
%#loc:@DeepHit/fully_connected/biases*
	container *
dtype0*
shape:
*
shared_name 
ж
%DeepHit/fully_connected/biases/AssignAssignDeepHit/fully_connected/biases0DeepHit/fully_connected/biases/Initializer/zeros*
T0*1
_class'
%#loc:@DeepHit/fully_connected/biases*
use_locking(*
validate_shape(
Л
#DeepHit/fully_connected/biases/readIdentityDeepHit/fully_connected/biases*
T0*1
_class'
%#loc:@DeepHit/fully_connected/biases
Н
DeepHit/fully_connected/MatMulMatMulDeepHit/inputs$DeepHit/fully_connected/weights/read*
T0*
transpose_a( *
transpose_b( 
П
DeepHit/fully_connected/BiasAddBiasAddDeepHit/fully_connected/MatMul#DeepHit/fully_connected/biases/read*
T0*
data_formatNHWC
L
DeepHit/fully_connected/EluEluDeepHit/fully_connected/BiasAdd*
T0
:
DeepHit/sub/xConst*
dtype0*
valueB
 *  А?
D
DeepHit/subSubDeepHit/sub/xDeepHit/keep_probability*
T0
B
DeepHit/dropout/ConstConst*
dtype0*
valueB
 *  А?
G
DeepHit/dropout/SubSubDeepHit/dropout/ConstDeepHit/sub*
T0
]
DeepHit/dropout/RealDivRealDivDeepHit/fully_connected/EluDeepHit/dropout/Sub*
T0
T
DeepHit/dropout/ShapeShapeDeepHit/fully_connected/Elu*
T0*
out_type0
Г
,DeepHit/dropout/random_uniform/RandomUniformRandomUniformDeepHit/dropout/Shape*
T0*
dtype0*

seed *
seed2 
p
DeepHit/dropout/GreaterEqualGreaterEqual,DeepHit/dropout/random_uniform/RandomUniformDeepHit/sub*
T0
b
DeepHit/dropout/CastCastDeepHit/dropout/GreaterEqual*

DstT0*

SrcT0
*
Truncate( 
R
DeepHit/dropout/MulMulDeepHit/dropout/RealDivDeepHit/dropout/Cast*
T0
ѓ
DDeepHit/fully_connected_1/weights/Initializer/truncated_normal/shapeConst*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
dtype0*
valueB"
   
   
¶
CDeepHit/fully_connected_1/weights/Initializer/truncated_normal/meanConst*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
dtype0*
valueB
 *    
®
EDeepHit/fully_connected_1/weights/Initializer/truncated_normal/stddevConst*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
dtype0*
valueB
 *ЉЄ>
М
NDeepHit/fully_connected_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalDDeepHit/fully_connected_1/weights/Initializer/truncated_normal/shape*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
dtype0*

seed *
seed2 
Я
BDeepHit/fully_connected_1/weights/Initializer/truncated_normal/mulMulNDeepHit/fully_connected_1/weights/Initializer/truncated_normal/TruncatedNormalEDeepHit/fully_connected_1/weights/Initializer/truncated_normal/stddev*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights
П
>DeepHit/fully_connected_1/weights/Initializer/truncated_normalAddV2BDeepHit/fully_connected_1/weights/Initializer/truncated_normal/mulCDeepHit/fully_connected_1/weights/Initializer/truncated_normal/mean*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights
Ђ
!DeepHit/fully_connected_1/weights
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
	container *
dtype0*
shape
:

*
shared_name 
э
(DeepHit/fully_connected_1/weights/AssignAssign!DeepHit/fully_connected_1/weights>DeepHit/fully_connected_1/weights/Initializer/truncated_normal*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
use_locking(*
validate_shape(
Ф
&DeepHit/fully_connected_1/weights/readIdentity!DeepHit/fully_connected_1/weights*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights
n
3DeepHit/fully_connected_1/kernel/Regularizer/SquareSquare&DeepHit/fully_connected_1/weights/read*
T0
g
2DeepHit/fully_connected_1/kernel/Regularizer/ConstConst*
dtype0*
valueB"       
∆
0DeepHit/fully_connected_1/kernel/Regularizer/SumSum3DeepHit/fully_connected_1/kernel/Regularizer/Square2DeepHit/fully_connected_1/kernel/Regularizer/Const*
T0*

Tidx0*
	keep_dims( 
_
2DeepHit/fully_connected_1/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *ЈQ8
¶
0DeepHit/fully_connected_1/kernel/Regularizer/mulMul2DeepHit/fully_connected_1/kernel/Regularizer/mul/x0DeepHit/fully_connected_1/kernel/Regularizer/Sum*
T0
Ш
2DeepHit/fully_connected_1/biases/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
dtype0*
valueB
*    
•
 DeepHit/fully_connected_1/biases
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
	container *
dtype0*
shape:
*
shared_name 
о
'DeepHit/fully_connected_1/biases/AssignAssign DeepHit/fully_connected_1/biases2DeepHit/fully_connected_1/biases/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
use_locking(*
validate_shape(
С
%DeepHit/fully_connected_1/biases/readIdentity DeepHit/fully_connected_1/biases*
T0*3
_class)
'%loc:@DeepHit/fully_connected_1/biases
Ц
 DeepHit/fully_connected_1/MatMulMatMulDeepHit/dropout/Mul&DeepHit/fully_connected_1/weights/read*
T0*
transpose_a( *
transpose_b( 
Х
!DeepHit/fully_connected_1/BiasAddBiasAdd DeepHit/fully_connected_1/MatMul%DeepHit/fully_connected_1/biases/read*
T0*
data_formatNHWC
P
DeepHit/fully_connected_1/EluElu!DeepHit/fully_connected_1/BiasAdd*
T0
<
DeepHit/sub_1/xConst*
dtype0*
valueB
 *  А?
H
DeepHit/sub_1SubDeepHit/sub_1/xDeepHit/keep_probability*
T0
D
DeepHit/dropout_1/ConstConst*
dtype0*
valueB
 *  А?
M
DeepHit/dropout_1/SubSubDeepHit/dropout_1/ConstDeepHit/sub_1*
T0
c
DeepHit/dropout_1/RealDivRealDivDeepHit/fully_connected_1/EluDeepHit/dropout_1/Sub*
T0
X
DeepHit/dropout_1/ShapeShapeDeepHit/fully_connected_1/Elu*
T0*
out_type0
З
.DeepHit/dropout_1/random_uniform/RandomUniformRandomUniformDeepHit/dropout_1/Shape*
T0*
dtype0*

seed *
seed2 
v
DeepHit/dropout_1/GreaterEqualGreaterEqual.DeepHit/dropout_1/random_uniform/RandomUniformDeepHit/sub_1*
T0
f
DeepHit/dropout_1/CastCastDeepHit/dropout_1/GreaterEqual*

DstT0*

SrcT0
*
Truncate( 
X
DeepHit/dropout_1/MulMulDeepHit/dropout_1/RealDivDeepHit/dropout_1/Cast*
T0
ѓ
DDeepHit/fully_connected_2/weights/Initializer/truncated_normal/shapeConst*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
dtype0*
valueB"
   
   
¶
CDeepHit/fully_connected_2/weights/Initializer/truncated_normal/meanConst*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
dtype0*
valueB
 *    
®
EDeepHit/fully_connected_2/weights/Initializer/truncated_normal/stddevConst*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
dtype0*
valueB
 *ЉЄ>
М
NDeepHit/fully_connected_2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalDDeepHit/fully_connected_2/weights/Initializer/truncated_normal/shape*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
dtype0*

seed *
seed2 
Я
BDeepHit/fully_connected_2/weights/Initializer/truncated_normal/mulMulNDeepHit/fully_connected_2/weights/Initializer/truncated_normal/TruncatedNormalEDeepHit/fully_connected_2/weights/Initializer/truncated_normal/stddev*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights
П
>DeepHit/fully_connected_2/weights/Initializer/truncated_normalAddV2BDeepHit/fully_connected_2/weights/Initializer/truncated_normal/mulCDeepHit/fully_connected_2/weights/Initializer/truncated_normal/mean*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights
Ђ
!DeepHit/fully_connected_2/weights
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
	container *
dtype0*
shape
:

*
shared_name 
э
(DeepHit/fully_connected_2/weights/AssignAssign!DeepHit/fully_connected_2/weights>DeepHit/fully_connected_2/weights/Initializer/truncated_normal*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
use_locking(*
validate_shape(
Ф
&DeepHit/fully_connected_2/weights/readIdentity!DeepHit/fully_connected_2/weights*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights
n
3DeepHit/fully_connected_2/kernel/Regularizer/SquareSquare&DeepHit/fully_connected_2/weights/read*
T0
g
2DeepHit/fully_connected_2/kernel/Regularizer/ConstConst*
dtype0*
valueB"       
∆
0DeepHit/fully_connected_2/kernel/Regularizer/SumSum3DeepHit/fully_connected_2/kernel/Regularizer/Square2DeepHit/fully_connected_2/kernel/Regularizer/Const*
T0*

Tidx0*
	keep_dims( 
_
2DeepHit/fully_connected_2/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *ЈQ8
¶
0DeepHit/fully_connected_2/kernel/Regularizer/mulMul2DeepHit/fully_connected_2/kernel/Regularizer/mul/x0DeepHit/fully_connected_2/kernel/Regularizer/Sum*
T0
Ш
2DeepHit/fully_connected_2/biases/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
dtype0*
valueB
*    
•
 DeepHit/fully_connected_2/biases
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
	container *
dtype0*
shape:
*
shared_name 
о
'DeepHit/fully_connected_2/biases/AssignAssign DeepHit/fully_connected_2/biases2DeepHit/fully_connected_2/biases/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
use_locking(*
validate_shape(
С
%DeepHit/fully_connected_2/biases/readIdentity DeepHit/fully_connected_2/biases*
T0*3
_class)
'%loc:@DeepHit/fully_connected_2/biases
Ш
 DeepHit/fully_connected_2/MatMulMatMulDeepHit/dropout_1/Mul&DeepHit/fully_connected_2/weights/read*
T0*
transpose_a( *
transpose_b( 
Х
!DeepHit/fully_connected_2/BiasAddBiasAdd DeepHit/fully_connected_2/MatMul%DeepHit/fully_connected_2/biases/read*
T0*
data_formatNHWC
P
DeepHit/fully_connected_2/EluElu!DeepHit/fully_connected_2/BiasAdd*
T0
=
DeepHit/concat/axisConst*
dtype0*
value	B :
|
DeepHit/concatConcatV2DeepHit/inputsDeepHit/fully_connected_2/EluDeepHit/concat/axis*
N*
T0*

Tidx0
ѓ
DDeepHit/fully_connected_3/weights/Initializer/truncated_normal/shapeConst*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
dtype0*
valueB"j      
¶
CDeepHit/fully_connected_3/weights/Initializer/truncated_normal/meanConst*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
dtype0*
valueB
 *    
®
EDeepHit/fully_connected_3/weights/Initializer/truncated_normal/stddevConst*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
dtype0*
valueB
 *≥™>
М
NDeepHit/fully_connected_3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalDDeepHit/fully_connected_3/weights/Initializer/truncated_normal/shape*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
dtype0*

seed *
seed2 
Я
BDeepHit/fully_connected_3/weights/Initializer/truncated_normal/mulMulNDeepHit/fully_connected_3/weights/Initializer/truncated_normal/TruncatedNormalEDeepHit/fully_connected_3/weights/Initializer/truncated_normal/stddev*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights
П
>DeepHit/fully_connected_3/weights/Initializer/truncated_normalAddV2BDeepHit/fully_connected_3/weights/Initializer/truncated_normal/mulCDeepHit/fully_connected_3/weights/Initializer/truncated_normal/mean*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights
Ђ
!DeepHit/fully_connected_3/weights
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
	container *
dtype0*
shape
:j*
shared_name 
э
(DeepHit/fully_connected_3/weights/AssignAssign!DeepHit/fully_connected_3/weights>DeepHit/fully_connected_3/weights/Initializer/truncated_normal*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
use_locking(*
validate_shape(
Ф
&DeepHit/fully_connected_3/weights/readIdentity!DeepHit/fully_connected_3/weights*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights
n
3DeepHit/fully_connected_3/kernel/Regularizer/SquareSquare&DeepHit/fully_connected_3/weights/read*
T0
g
2DeepHit/fully_connected_3/kernel/Regularizer/ConstConst*
dtype0*
valueB"       
∆
0DeepHit/fully_connected_3/kernel/Regularizer/SumSum3DeepHit/fully_connected_3/kernel/Regularizer/Square2DeepHit/fully_connected_3/kernel/Regularizer/Const*
T0*

Tidx0*
	keep_dims( 
_
2DeepHit/fully_connected_3/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *ЈQ8
¶
0DeepHit/fully_connected_3/kernel/Regularizer/mulMul2DeepHit/fully_connected_3/kernel/Regularizer/mul/x0DeepHit/fully_connected_3/kernel/Regularizer/Sum*
T0
Ш
2DeepHit/fully_connected_3/biases/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
dtype0*
valueB*    
•
 DeepHit/fully_connected_3/biases
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
	container *
dtype0*
shape:*
shared_name 
о
'DeepHit/fully_connected_3/biases/AssignAssign DeepHit/fully_connected_3/biases2DeepHit/fully_connected_3/biases/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
use_locking(*
validate_shape(
С
%DeepHit/fully_connected_3/biases/readIdentity DeepHit/fully_connected_3/biases*
T0*3
_class)
'%loc:@DeepHit/fully_connected_3/biases
С
 DeepHit/fully_connected_3/MatMulMatMulDeepHit/concat&DeepHit/fully_connected_3/weights/read*
T0*
transpose_a( *
transpose_b( 
Х
!DeepHit/fully_connected_3/BiasAddBiasAdd DeepHit/fully_connected_3/MatMul%DeepHit/fully_connected_3/biases/read*
T0*
data_formatNHWC
P
DeepHit/fully_connected_3/EluElu!DeepHit/fully_connected_3/BiasAdd*
T0
ѓ
DDeepHit/fully_connected_4/weights/Initializer/truncated_normal/shapeConst*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
dtype0*
valueB"j      
¶
CDeepHit/fully_connected_4/weights/Initializer/truncated_normal/meanConst*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
dtype0*
valueB
 *    
®
EDeepHit/fully_connected_4/weights/Initializer/truncated_normal/stddevConst*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
dtype0*
valueB
 *≥™>
М
NDeepHit/fully_connected_4/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalDDeepHit/fully_connected_4/weights/Initializer/truncated_normal/shape*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
dtype0*

seed *
seed2 
Я
BDeepHit/fully_connected_4/weights/Initializer/truncated_normal/mulMulNDeepHit/fully_connected_4/weights/Initializer/truncated_normal/TruncatedNormalEDeepHit/fully_connected_4/weights/Initializer/truncated_normal/stddev*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights
П
>DeepHit/fully_connected_4/weights/Initializer/truncated_normalAddV2BDeepHit/fully_connected_4/weights/Initializer/truncated_normal/mulCDeepHit/fully_connected_4/weights/Initializer/truncated_normal/mean*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights
Ђ
!DeepHit/fully_connected_4/weights
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
	container *
dtype0*
shape
:j*
shared_name 
э
(DeepHit/fully_connected_4/weights/AssignAssign!DeepHit/fully_connected_4/weights>DeepHit/fully_connected_4/weights/Initializer/truncated_normal*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
use_locking(*
validate_shape(
Ф
&DeepHit/fully_connected_4/weights/readIdentity!DeepHit/fully_connected_4/weights*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights
n
3DeepHit/fully_connected_4/kernel/Regularizer/SquareSquare&DeepHit/fully_connected_4/weights/read*
T0
g
2DeepHit/fully_connected_4/kernel/Regularizer/ConstConst*
dtype0*
valueB"       
∆
0DeepHit/fully_connected_4/kernel/Regularizer/SumSum3DeepHit/fully_connected_4/kernel/Regularizer/Square2DeepHit/fully_connected_4/kernel/Regularizer/Const*
T0*

Tidx0*
	keep_dims( 
_
2DeepHit/fully_connected_4/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *ЈQ8
¶
0DeepHit/fully_connected_4/kernel/Regularizer/mulMul2DeepHit/fully_connected_4/kernel/Regularizer/mul/x0DeepHit/fully_connected_4/kernel/Regularizer/Sum*
T0
Ш
2DeepHit/fully_connected_4/biases/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
dtype0*
valueB*    
•
 DeepHit/fully_connected_4/biases
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
	container *
dtype0*
shape:*
shared_name 
о
'DeepHit/fully_connected_4/biases/AssignAssign DeepHit/fully_connected_4/biases2DeepHit/fully_connected_4/biases/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
use_locking(*
validate_shape(
С
%DeepHit/fully_connected_4/biases/readIdentity DeepHit/fully_connected_4/biases*
T0*3
_class)
'%loc:@DeepHit/fully_connected_4/biases
С
 DeepHit/fully_connected_4/MatMulMatMulDeepHit/concat&DeepHit/fully_connected_4/weights/read*
T0*
transpose_a( *
transpose_b( 
Х
!DeepHit/fully_connected_4/BiasAddBiasAdd DeepHit/fully_connected_4/MatMul%DeepHit/fully_connected_4/biases/read*
T0*
data_formatNHWC
P
DeepHit/fully_connected_4/EluElu!DeepHit/fully_connected_4/BiasAdd*
T0
q
DeepHit/stackPackDeepHit/fully_connected_3/EluDeepHit/fully_connected_4/Elu*
N*
T0*

axis
J
DeepHit/Reshape/shapeConst*
dtype0*
valueB"€€€€(   
W
DeepHit/ReshapeReshapeDeepHit/stackDeepHit/Reshape/shape*
T0*
Tshape0
<
DeepHit/sub_2/xConst*
dtype0*
valueB
 *  А?
H
DeepHit/sub_2SubDeepHit/sub_2/xDeepHit/keep_probability*
T0
D
DeepHit/dropout_2/ConstConst*
dtype0*
valueB
 *  А?
M
DeepHit/dropout_2/SubSubDeepHit/dropout_2/ConstDeepHit/sub_2*
T0
U
DeepHit/dropout_2/RealDivRealDivDeepHit/ReshapeDeepHit/dropout_2/Sub*
T0
J
DeepHit/dropout_2/ShapeShapeDeepHit/Reshape*
T0*
out_type0
З
.DeepHit/dropout_2/random_uniform/RandomUniformRandomUniformDeepHit/dropout_2/Shape*
T0*
dtype0*

seed *
seed2 
v
DeepHit/dropout_2/GreaterEqualGreaterEqual.DeepHit/dropout_2/random_uniform/RandomUniformDeepHit/sub_2*
T0
f
DeepHit/dropout_2/CastCastDeepHit/dropout_2/GreaterEqual*

DstT0*

SrcT0
*
Truncate( 
X
DeepHit/dropout_2/MulMulDeepHit/dropout_2/RealDivDeepHit/dropout_2/Cast*
T0
Щ
9DeepHit/Output/weights/Initializer/truncated_normal/shapeConst*)
_class
loc:@DeepHit/Output/weights*
dtype0*
valueB"(     
Р
8DeepHit/Output/weights/Initializer/truncated_normal/meanConst*)
_class
loc:@DeepHit/Output/weights*
dtype0*
valueB
 *    
Т
:DeepHit/Output/weights/Initializer/truncated_normal/stddevConst*)
_class
loc:@DeepHit/Output/weights*
dtype0*
valueB
 *]ґ=
л
CDeepHit/Output/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9DeepHit/Output/weights/Initializer/truncated_normal/shape*
T0*)
_class
loc:@DeepHit/Output/weights*
dtype0*

seed *
seed2 
у
7DeepHit/Output/weights/Initializer/truncated_normal/mulMulCDeepHit/Output/weights/Initializer/truncated_normal/TruncatedNormal:DeepHit/Output/weights/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@DeepHit/Output/weights
г
3DeepHit/Output/weights/Initializer/truncated_normalAddV27DeepHit/Output/weights/Initializer/truncated_normal/mul8DeepHit/Output/weights/Initializer/truncated_normal/mean*
T0*)
_class
loc:@DeepHit/Output/weights
Ц
DeepHit/Output/weights
VariableV2*)
_class
loc:@DeepHit/Output/weights*
	container *
dtype0*
shape:	(Ю*
shared_name 
—
DeepHit/Output/weights/AssignAssignDeepHit/Output/weights3DeepHit/Output/weights/Initializer/truncated_normal*
T0*)
_class
loc:@DeepHit/Output/weights*
use_locking(*
validate_shape(
s
DeepHit/Output/weights/readIdentityDeepHit/Output/weights*
T0*)
_class
loc:@DeepHit/Output/weights
R
%DeepHit/Output/kernel/Regularizer/AbsAbsDeepHit/Output/weights/read*
T0
\
'DeepHit/Output/kernel/Regularizer/ConstConst*
dtype0*
valueB"       
Ґ
%DeepHit/Output/kernel/Regularizer/SumSum%DeepHit/Output/kernel/Regularizer/Abs'DeepHit/Output/kernel/Regularizer/Const*
T0*

Tidx0*
	keep_dims( 
T
'DeepHit/Output/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *Ј—8
Е
%DeepHit/Output/kernel/Regularizer/mulMul'DeepHit/Output/kernel/Regularizer/mul/x%DeepHit/Output/kernel/Regularizer/Sum*
T0
Г
'DeepHit/Output/biases/Initializer/zerosConst*(
_class
loc:@DeepHit/Output/biases*
dtype0*
valueBЮ*    
Р
DeepHit/Output/biases
VariableV2*(
_class
loc:@DeepHit/Output/biases*
	container *
dtype0*
shape:Ю*
shared_name 
¬
DeepHit/Output/biases/AssignAssignDeepHit/Output/biases'DeepHit/Output/biases/Initializer/zeros*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking(*
validate_shape(
p
DeepHit/Output/biases/readIdentityDeepHit/Output/biases*
T0*(
_class
loc:@DeepHit/Output/biases
В
DeepHit/Output/MatMulMatMulDeepHit/dropout_2/MulDeepHit/Output/weights/read*
T0*
transpose_a( *
transpose_b( 
t
DeepHit/Output/BiasAddBiasAddDeepHit/Output/MatMulDeepHit/Output/biases/read*
T0*
data_formatNHWC
B
DeepHit/Output/SoftmaxSoftmaxDeepHit/Output/BiasAdd*
T0
P
DeepHit/Reshape_1/shapeConst*
dtype0*!
valueB"€€€€   П   
d
DeepHit/Reshape_1ReshapeDeepHit/Output/SoftmaxDeepHit/Reshape_1/shape*
T0*
Tshape0
-
DeepHit/SignSignDeepHit/labels*
T0
=
DeepHit/mulMulDeepHit/mask1DeepHit/Reshape_1*
T0
G
DeepHit/Sum/reduction_indicesConst*
dtype0*
value	B :
d
DeepHit/SumSumDeepHit/mulDeepHit/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
I
DeepHit/Sum_1/reduction_indicesConst*
dtype0*
value	B :
h
DeepHit/Sum_1SumDeepHit/SumDeepHit/Sum_1/reduction_indices*
T0*

Tidx0*
	keep_dims(
:
DeepHit/add/yConst*
dtype0*
valueB
 *wћ+2
;
DeepHit/addAddV2DeepHit/Sum_1DeepHit/add/y*
T0
(
DeepHit/LogLogDeepHit/add*
T0
8
DeepHit/mul_1MulDeepHit/SignDeepHit/Log*
T0
?
DeepHit/mul_2MulDeepHit/mask1DeepHit/Reshape_1*
T0
I
DeepHit/Sum_2/reduction_indicesConst*
dtype0*
value	B :
j
DeepHit/Sum_2SumDeepHit/mul_2DeepHit/Sum_2/reduction_indices*
T0*

Tidx0*
	keep_dims( 
I
DeepHit/Sum_3/reduction_indicesConst*
dtype0*
value	B :
j
DeepHit/Sum_3SumDeepHit/Sum_2DeepHit/Sum_3/reduction_indices*
T0*

Tidx0*
	keep_dims(
<
DeepHit/sub_3/xConst*
dtype0*
valueB
 *  А?
<
DeepHit/sub_3SubDeepHit/sub_3/xDeepHit/Sign*
T0
<
DeepHit/add_1/yConst*
dtype0*
valueB
 *wћ+2
?
DeepHit/add_1AddV2DeepHit/Sum_3DeepHit/add_1/y*
T0
,
DeepHit/Log_1LogDeepHit/add_1*
T0
;
DeepHit/mul_3MulDeepHit/sub_3DeepHit/Log_1*
T0
<
DeepHit/mul_4/xConst*
dtype0*
valueB
 *  А?
=
DeepHit/mul_4MulDeepHit/mul_4/xDeepHit/mul_3*
T0
=
DeepHit/add_2AddV2DeepHit/mul_1DeepHit/mul_4*
T0
B
DeepHit/ConstConst*
dtype0*
valueB"       
X
DeepHit/MeanMeanDeepHit/add_2DeepHit/Const*
T0*

Tidx0*
	keep_dims( 
)
DeepHit/NegNegDeepHit/Mean*
T0
<
DeepHit/Const_1Const*
dtype0*
valueB
 *Ќћћ=
O
DeepHit/ones_like/ShapeShapeDeepHit/timetoevents*
T0*
out_type0
D
DeepHit/ones_like/ConstConst*
dtype0*
valueB
 *  А?
f
DeepHit/ones_likeFillDeepHit/ones_like/ShapeDeepHit/ones_like/Const*
T0*

index_type0
<
DeepHit/Equal/yConst*
dtype0*
valueB
 *  А?
`
DeepHit/EqualEqualDeepHit/labelsDeepHit/Equal/y*
T0*
incompatible_shape_error(
K
DeepHit/CastCastDeepHit/Equal*

DstT0*

SrcT0
*
Truncate( 
E
DeepHit/SqueezeSqueezeDeepHit/Cast*
T0*
squeeze_dims
 
.
DeepHit/DiagDiagDeepHit/Squeeze*
T0
L
DeepHit/Slice/beginConst*
dtype0*!
valueB"            
K
DeepHit/Slice/sizeConst*
dtype0*!
valueB"€€€€   €€€€
h
DeepHit/SliceSliceDeepHit/Reshape_1DeepHit/Slice/beginDeepHit/Slice/size*
Index0*
T0
L
DeepHit/Reshape_2/shapeConst*
dtype0*
valueB"€€€€П   
[
DeepHit/Reshape_2ReshapeDeepHit/SliceDeepHit/Reshape_2/shape*
T0*
Tshape0
K
DeepHit/transpose/permConst*
dtype0*
valueB"       
[
DeepHit/transpose	TransposeDeepHit/mask2DeepHit/transpose/perm*
T0*
Tperm0
m
DeepHit/MatMulMatMulDeepHit/Reshape_2DeepHit/transpose*
T0*
transpose_a( *
transpose_b( 
5
DeepHit/DiagPartDiagPartDeepHit/MatMul*
T0
L
DeepHit/Reshape_3/shapeConst*
dtype0*
valueB"€€€€   
^
DeepHit/Reshape_3ReshapeDeepHit/DiagPartDeepHit/Reshape_3/shape*
T0*
Tshape0
M
DeepHit/transpose_1/permConst*
dtype0*
valueB"       
c
DeepHit/transpose_1	TransposeDeepHit/Reshape_3DeepHit/transpose_1/perm*
T0*
Tperm0
q
DeepHit/MatMul_1MatMulDeepHit/ones_likeDeepHit/transpose_1*
T0*
transpose_a( *
transpose_b( 
?
DeepHit/sub_4SubDeepHit/MatMul_1DeepHit/MatMul*
T0
M
DeepHit/transpose_2/permConst*
dtype0*
valueB"       
_
DeepHit/transpose_2	TransposeDeepHit/sub_4DeepHit/transpose_2/perm*
T0*
Tperm0
M
DeepHit/transpose_3/permConst*
dtype0*
valueB"       
f
DeepHit/transpose_3	TransposeDeepHit/timetoeventsDeepHit/transpose_3/perm*
T0*
Tperm0
q
DeepHit/MatMul_2MatMulDeepHit/ones_likeDeepHit/transpose_3*
T0*
transpose_a( *
transpose_b( 
M
DeepHit/transpose_4/permConst*
dtype0*
valueB"       
c
DeepHit/transpose_4	TransposeDeepHit/ones_likeDeepHit/transpose_4/perm*
T0*
Tperm0
t
DeepHit/MatMul_3MatMulDeepHit/timetoeventsDeepHit/transpose_4*
T0*
transpose_a( *
transpose_b( 
A
DeepHit/sub_5SubDeepHit/MatMul_2DeepHit/MatMul_3*
T0
.
DeepHit/Sign_1SignDeepHit/sub_5*
T0
-
DeepHit/ReluReluDeepHit/Sign_1*
T0
`
DeepHit/MatMul_4BatchMatMulV2DeepHit/DiagDeepHit/Relu*
T0*
adj_x( *
adj_y( 
2
DeepHit/Neg_1NegDeepHit/transpose_2*
T0
C
DeepHit/truedivRealDivDeepHit/Neg_1DeepHit/Const_1*
T0
,
DeepHit/ExpExpDeepHit/truediv*
T0
<
DeepHit/mul_5MulDeepHit/MatMul_4DeepHit/Exp*
T0
J
 DeepHit/Mean_1/reduction_indicesConst*
dtype0*
value	B :
m
DeepHit/Mean_1MeanDeepHit/mul_5 DeepHit/Mean_1/reduction_indices*
T0*

Tidx0*
	keep_dims(
Q
DeepHit/ones_like_1/ShapeShapeDeepHit/timetoevents*
T0*
out_type0
F
DeepHit/ones_like_1/ConstConst*
dtype0*
valueB
 *  А?
l
DeepHit/ones_like_1FillDeepHit/ones_like_1/ShapeDeepHit/ones_like_1/Const*
T0*

index_type0
>
DeepHit/Equal_1/yConst*
dtype0*
valueB
 *   @
d
DeepHit/Equal_1EqualDeepHit/labelsDeepHit/Equal_1/y*
T0*
incompatible_shape_error(
O
DeepHit/Cast_1CastDeepHit/Equal_1*

DstT0*

SrcT0
*
Truncate( 
I
DeepHit/Squeeze_1SqueezeDeepHit/Cast_1*
T0*
squeeze_dims
 
2
DeepHit/Diag_1DiagDeepHit/Squeeze_1*
T0
N
DeepHit/Slice_1/beginConst*
dtype0*!
valueB"           
M
DeepHit/Slice_1/sizeConst*
dtype0*!
valueB"€€€€   €€€€
n
DeepHit/Slice_1SliceDeepHit/Reshape_1DeepHit/Slice_1/beginDeepHit/Slice_1/size*
Index0*
T0
L
DeepHit/Reshape_4/shapeConst*
dtype0*
valueB"€€€€П   
]
DeepHit/Reshape_4ReshapeDeepHit/Slice_1DeepHit/Reshape_4/shape*
T0*
Tshape0
M
DeepHit/transpose_5/permConst*
dtype0*
valueB"       
_
DeepHit/transpose_5	TransposeDeepHit/mask2DeepHit/transpose_5/perm*
T0*
Tperm0
q
DeepHit/MatMul_5MatMulDeepHit/Reshape_4DeepHit/transpose_5*
T0*
transpose_a( *
transpose_b( 
9
DeepHit/DiagPart_1DiagPartDeepHit/MatMul_5*
T0
L
DeepHit/Reshape_5/shapeConst*
dtype0*
valueB"€€€€   
`
DeepHit/Reshape_5ReshapeDeepHit/DiagPart_1DeepHit/Reshape_5/shape*
T0*
Tshape0
M
DeepHit/transpose_6/permConst*
dtype0*
valueB"       
c
DeepHit/transpose_6	TransposeDeepHit/Reshape_5DeepHit/transpose_6/perm*
T0*
Tperm0
s
DeepHit/MatMul_6MatMulDeepHit/ones_like_1DeepHit/transpose_6*
T0*
transpose_a( *
transpose_b( 
A
DeepHit/sub_6SubDeepHit/MatMul_6DeepHit/MatMul_5*
T0
M
DeepHit/transpose_7/permConst*
dtype0*
valueB"       
_
DeepHit/transpose_7	TransposeDeepHit/sub_6DeepHit/transpose_7/perm*
T0*
Tperm0
M
DeepHit/transpose_8/permConst*
dtype0*
valueB"       
f
DeepHit/transpose_8	TransposeDeepHit/timetoeventsDeepHit/transpose_8/perm*
T0*
Tperm0
s
DeepHit/MatMul_7MatMulDeepHit/ones_like_1DeepHit/transpose_8*
T0*
transpose_a( *
transpose_b( 
M
DeepHit/transpose_9/permConst*
dtype0*
valueB"       
e
DeepHit/transpose_9	TransposeDeepHit/ones_like_1DeepHit/transpose_9/perm*
T0*
Tperm0
t
DeepHit/MatMul_8MatMulDeepHit/timetoeventsDeepHit/transpose_9*
T0*
transpose_a( *
transpose_b( 
A
DeepHit/sub_7SubDeepHit/MatMul_7DeepHit/MatMul_8*
T0
.
DeepHit/Sign_2SignDeepHit/sub_7*
T0
/
DeepHit/Relu_1ReluDeepHit/Sign_2*
T0
d
DeepHit/MatMul_9BatchMatMulV2DeepHit/Diag_1DeepHit/Relu_1*
T0*
adj_x( *
adj_y( 
2
DeepHit/Neg_2NegDeepHit/transpose_7*
T0
E
DeepHit/truediv_1RealDivDeepHit/Neg_2DeepHit/Const_1*
T0
0
DeepHit/Exp_1ExpDeepHit/truediv_1*
T0
>
DeepHit/mul_6MulDeepHit/MatMul_9DeepHit/Exp_1*
T0
J
 DeepHit/Mean_2/reduction_indicesConst*
dtype0*
value	B :
m
DeepHit/Mean_2MeanDeepHit/mul_6 DeepHit/Mean_2/reduction_indices*
T0*

Tidx0*
	keep_dims(
U
DeepHit/stack_1PackDeepHit/Mean_1DeepHit/Mean_2*
N*
T0*

axis
L
DeepHit/Reshape_6/shapeConst*
dtype0*
valueB"€€€€   
]
DeepHit/Reshape_6ReshapeDeepHit/stack_1DeepHit/Reshape_6/shape*
T0*
Tshape0
J
 DeepHit/Mean_3/reduction_indicesConst*
dtype0*
value	B :
q
DeepHit/Mean_3MeanDeepHit/Reshape_6 DeepHit/Mean_3/reduction_indices*
T0*

Tidx0*
	keep_dims(
D
DeepHit/Const_2Const*
dtype0*
valueB"       
[
DeepHit/Sum_4SumDeepHit/Mean_3DeepHit/Const_2*
T0*

Tidx0*
	keep_dims( 
Q
DeepHit/ones_like_2/ShapeShapeDeepHit/timetoevents*
T0*
out_type0
F
DeepHit/ones_like_2/ConstConst*
dtype0*
valueB
 *  А?
l
DeepHit/ones_like_2FillDeepHit/ones_like_2/ShapeDeepHit/ones_like_2/Const*
T0*

index_type0
>
DeepHit/Equal_2/yConst*
dtype0*
valueB
 *  А?
d
DeepHit/Equal_2EqualDeepHit/labelsDeepHit/Equal_2/y*
T0*
incompatible_shape_error(
O
DeepHit/Cast_2CastDeepHit/Equal_2*

DstT0*

SrcT0
*
Truncate( 
N
DeepHit/Slice_2/beginConst*
dtype0*!
valueB"            
M
DeepHit/Slice_2/sizeConst*
dtype0*!
valueB"€€€€   €€€€
n
DeepHit/Slice_2SliceDeepHit/Reshape_1DeepHit/Slice_2/beginDeepHit/Slice_2/size*
Index0*
T0
L
DeepHit/Reshape_7/shapeConst*
dtype0*
valueB"€€€€П   
]
DeepHit/Reshape_7ReshapeDeepHit/Slice_2DeepHit/Reshape_7/shape*
T0*
Tshape0
?
DeepHit/mul_7MulDeepHit/Reshape_7DeepHit/mask2*
T0
I
DeepHit/Sum_5/reduction_indicesConst*
dtype0*
value	B : 
j
DeepHit/Sum_5SumDeepHit/mul_7DeepHit/Sum_5/reduction_indices*
T0*

Tidx0*
	keep_dims( 
<
DeepHit/sub_8SubDeepHit/Sum_5DeepHit/Cast_2*
T0
:
DeepHit/pow/yConst*
dtype0*
valueB
 *   @
9
DeepHit/powPowDeepHit/sub_8DeepHit/pow/y*
T0
J
 DeepHit/Mean_4/reduction_indicesConst*
dtype0*
value	B :
k
DeepHit/Mean_4MeanDeepHit/pow DeepHit/Mean_4/reduction_indices*
T0*

Tidx0*
	keep_dims(
Q
DeepHit/ones_like_3/ShapeShapeDeepHit/timetoevents*
T0*
out_type0
F
DeepHit/ones_like_3/ConstConst*
dtype0*
valueB
 *  А?
l
DeepHit/ones_like_3FillDeepHit/ones_like_3/ShapeDeepHit/ones_like_3/Const*
T0*

index_type0
>
DeepHit/Equal_3/yConst*
dtype0*
valueB
 *   @
d
DeepHit/Equal_3EqualDeepHit/labelsDeepHit/Equal_3/y*
T0*
incompatible_shape_error(
O
DeepHit/Cast_3CastDeepHit/Equal_3*

DstT0*

SrcT0
*
Truncate( 
N
DeepHit/Slice_3/beginConst*
dtype0*!
valueB"           
M
DeepHit/Slice_3/sizeConst*
dtype0*!
valueB"€€€€   €€€€
n
DeepHit/Slice_3SliceDeepHit/Reshape_1DeepHit/Slice_3/beginDeepHit/Slice_3/size*
Index0*
T0
L
DeepHit/Reshape_8/shapeConst*
dtype0*
valueB"€€€€П   
]
DeepHit/Reshape_8ReshapeDeepHit/Slice_3DeepHit/Reshape_8/shape*
T0*
Tshape0
?
DeepHit/mul_8MulDeepHit/Reshape_8DeepHit/mask2*
T0
I
DeepHit/Sum_6/reduction_indicesConst*
dtype0*
value	B : 
j
DeepHit/Sum_6SumDeepHit/mul_8DeepHit/Sum_6/reduction_indices*
T0*

Tidx0*
	keep_dims( 
<
DeepHit/sub_9SubDeepHit/Sum_6DeepHit/Cast_3*
T0
<
DeepHit/pow_1/yConst*
dtype0*
valueB
 *   @
=
DeepHit/pow_1PowDeepHit/sub_9DeepHit/pow_1/y*
T0
J
 DeepHit/Mean_5/reduction_indicesConst*
dtype0*
value	B :
m
DeepHit/Mean_5MeanDeepHit/pow_1 DeepHit/Mean_5/reduction_indices*
T0*

Tidx0*
	keep_dims(
U
DeepHit/stack_2PackDeepHit/Mean_4DeepHit/Mean_5*
N*
T0*

axis
L
DeepHit/Reshape_9/shapeConst*
dtype0*
valueB"€€€€   
]
DeepHit/Reshape_9ReshapeDeepHit/stack_2DeepHit/Reshape_9/shape*
T0*
Tshape0
J
 DeepHit/Mean_6/reduction_indicesConst*
dtype0*
value	B :
q
DeepHit/Mean_6MeanDeepHit/Reshape_9 DeepHit/Mean_6/reduction_indices*
T0*

Tidx0*
	keep_dims(
D
DeepHit/Const_3Const*
dtype0*
valueB"       
[
DeepHit/Sum_7SumDeepHit/Mean_6DeepHit/Const_3*
T0*

Tidx0*
	keep_dims( 
9
DeepHit/mul_9MulDeepHit/alphaDeepHit/Neg*
T0
;
DeepHit/mul_10MulDeepHit/betaDeepHit/Sum_4*
T0
>
DeepHit/add_3AddV2DeepHit/mul_9DeepHit/mul_10*
T0
<
DeepHit/mul_11MulDeepHit/gammaDeepHit/Sum_7*
T0
>
DeepHit/add_4AddV2DeepHit/add_3DeepHit/mul_11*
T0
Џ
!DeepHit/total_regularization_lossAddN.DeepHit/fully_connected/kernel/Regularizer/mul0DeepHit/fully_connected_1/kernel/Regularizer/mul0DeepHit/fully_connected_2/kernel/Regularizer/mul0DeepHit/fully_connected_3/kernel/Regularizer/mul0DeepHit/fully_connected_4/kernel/Regularizer/mul%DeepHit/Output/kernel/Regularizer/mul*
N*
T0
Q
DeepHit/add_5AddV2DeepHit/add_4!DeepHit/total_regularization_loss*
T0
@
DeepHit/gradients/ShapeConst*
dtype0*
valueB 
N
!DeepHit/gradients/grad_ys_0/ConstConst*
dtype0*
valueB
 *  А?
z
DeepHit/gradients/grad_ys_0FillDeepHit/gradients/Shape!DeepHit/gradients/grad_ys_0/Const*
T0*

index_type0
[
5DeepHit/gradients/DeepHit/add_5_grad/tuple/group_depsNoOp^DeepHit/gradients/grad_ys_0
„
=DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependencyIdentityDeepHit/gradients/grad_ys_06^DeepHit/gradients/DeepHit/add_5_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
ў
?DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1IdentityDeepHit/gradients/grad_ys_06^DeepHit/gradients/DeepHit/add_5_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
}
5DeepHit/gradients/DeepHit/add_4_grad/tuple/group_depsNoOp>^DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency
щ
=DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependencyIdentity=DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency6^DeepHit/gradients/DeepHit/add_4_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
ы
?DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency_1Identity=DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency6^DeepHit/gradients/DeepHit/add_4_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
У
IDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_depsNoOp@^DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1
£
QDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependencyIdentity?DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1J^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
•
SDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_1Identity?DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1J^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
•
SDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_2Identity?DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1J^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
•
SDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_3Identity?DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1J^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
•
SDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_4Identity?DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1J^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
•
SDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_5Identity?DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1J^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
}
5DeepHit/gradients/DeepHit/add_3_grad/tuple/group_depsNoOp>^DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency
щ
=DeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependencyIdentity=DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency6^DeepHit/gradients/DeepHit/add_3_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
ы
?DeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependency_1Identity=DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency6^DeepHit/gradients/DeepHit/add_3_grad/tuple/group_deps*
T0*.
_class$
" loc:@DeepHit/gradients/grad_ys_0
Й
)DeepHit/gradients/DeepHit/mul_11_grad/MulMul?DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency_1DeepHit/Sum_7*
T0
Л
+DeepHit/gradients/DeepHit/mul_11_grad/Mul_1Mul?DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency_1DeepHit/gamma*
T0
Ш
6DeepHit/gradients/DeepHit/mul_11_grad/tuple/group_depsNoOp*^DeepHit/gradients/DeepHit/mul_11_grad/Mul,^DeepHit/gradients/DeepHit/mul_11_grad/Mul_1
х
>DeepHit/gradients/DeepHit/mul_11_grad/tuple/control_dependencyIdentity)DeepHit/gradients/DeepHit/mul_11_grad/Mul7^DeepHit/gradients/DeepHit/mul_11_grad/tuple/group_deps*
T0*<
_class2
0.loc:@DeepHit/gradients/DeepHit/mul_11_grad/Mul
ы
@DeepHit/gradients/DeepHit/mul_11_grad/tuple/control_dependency_1Identity+DeepHit/gradients/DeepHit/mul_11_grad/Mul_17^DeepHit/gradients/DeepHit/mul_11_grad/tuple/group_deps*
T0*>
_class4
20loc:@DeepHit/gradients/DeepHit/mul_11_grad/Mul_1
№
IDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/MulMulQDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency.DeepHit/fully_connected/kernel/Regularizer/Sum*
T0
а
KDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_1MulQDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency0DeepHit/fully_connected/kernel/Regularizer/mul/x*
T0
ш
VDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/group_depsNoOpJ^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/MulL^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_1
х
^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityIDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/MulW^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul
ы
`DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityKDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_1W^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_1
в
KDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/MulMulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_10DeepHit/fully_connected_1/kernel/Regularizer/Sum*
T0
ж
MDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_1MulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_12DeepHit/fully_connected_1/kernel/Regularizer/mul/x*
T0
ю
XDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/group_depsNoOpL^DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/MulN^DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_1
э
`DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityKDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/MulY^DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul
Г
bDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityMDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_1Y^DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_1
в
KDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/MulMulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_20DeepHit/fully_connected_2/kernel/Regularizer/Sum*
T0
ж
MDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_1MulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_22DeepHit/fully_connected_2/kernel/Regularizer/mul/x*
T0
ю
XDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/group_depsNoOpL^DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/MulN^DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_1
э
`DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityKDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/MulY^DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul
Г
bDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityMDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_1Y^DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_1
в
KDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/MulMulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_30DeepHit/fully_connected_3/kernel/Regularizer/Sum*
T0
ж
MDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_1MulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_32DeepHit/fully_connected_3/kernel/Regularizer/mul/x*
T0
ю
XDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/group_depsNoOpL^DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/MulN^DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_1
э
`DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityKDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/MulY^DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul
Г
bDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityMDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_1Y^DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_1
в
KDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/MulMulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_40DeepHit/fully_connected_4/kernel/Regularizer/Sum*
T0
ж
MDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_1MulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_42DeepHit/fully_connected_4/kernel/Regularizer/mul/x*
T0
ю
XDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/group_depsNoOpL^DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/MulN^DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_1
э
`DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityKDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/MulY^DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul
Г
bDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityMDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_1Y^DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_1
ћ
@DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/MulMulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_5%DeepHit/Output/kernel/Regularizer/Sum*
T0
–
BDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_1MulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_5'DeepHit/Output/kernel/Regularizer/mul/x*
T0
Ё
MDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/group_depsNoOpA^DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/MulC^DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_1
—
UDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentity@DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/MulN^DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul
„
WDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityBDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_1N^DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_1
Д
(DeepHit/gradients/DeepHit/mul_9_grad/MulMul=DeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependencyDeepHit/Neg*
T0
И
*DeepHit/gradients/DeepHit/mul_9_grad/Mul_1Mul=DeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependencyDeepHit/alpha*
T0
Х
5DeepHit/gradients/DeepHit/mul_9_grad/tuple/group_depsNoOp)^DeepHit/gradients/DeepHit/mul_9_grad/Mul+^DeepHit/gradients/DeepHit/mul_9_grad/Mul_1
с
=DeepHit/gradients/DeepHit/mul_9_grad/tuple/control_dependencyIdentity(DeepHit/gradients/DeepHit/mul_9_grad/Mul6^DeepHit/gradients/DeepHit/mul_9_grad/tuple/group_deps*
T0*;
_class1
/-loc:@DeepHit/gradients/DeepHit/mul_9_grad/Mul
ч
?DeepHit/gradients/DeepHit/mul_9_grad/tuple/control_dependency_1Identity*DeepHit/gradients/DeepHit/mul_9_grad/Mul_16^DeepHit/gradients/DeepHit/mul_9_grad/tuple/group_deps*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/mul_9_grad/Mul_1
Й
)DeepHit/gradients/DeepHit/mul_10_grad/MulMul?DeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependency_1DeepHit/Sum_4*
T0
К
+DeepHit/gradients/DeepHit/mul_10_grad/Mul_1Mul?DeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependency_1DeepHit/beta*
T0
Ш
6DeepHit/gradients/DeepHit/mul_10_grad/tuple/group_depsNoOp*^DeepHit/gradients/DeepHit/mul_10_grad/Mul,^DeepHit/gradients/DeepHit/mul_10_grad/Mul_1
х
>DeepHit/gradients/DeepHit/mul_10_grad/tuple/control_dependencyIdentity)DeepHit/gradients/DeepHit/mul_10_grad/Mul7^DeepHit/gradients/DeepHit/mul_10_grad/tuple/group_deps*
T0*<
_class2
0.loc:@DeepHit/gradients/DeepHit/mul_10_grad/Mul
ы
@DeepHit/gradients/DeepHit/mul_10_grad/tuple/control_dependency_1Identity+DeepHit/gradients/DeepHit/mul_10_grad/Mul_17^DeepHit/gradients/DeepHit/mul_10_grad/tuple/group_deps*
T0*>
_class4
20loc:@DeepHit/gradients/DeepHit/mul_10_grad/Mul_1
g
2DeepHit/gradients/DeepHit/Sum_7_grad/Reshape/shapeConst*
dtype0*
valueB"      
ƒ
,DeepHit/gradients/DeepHit/Sum_7_grad/ReshapeReshape@DeepHit/gradients/DeepHit/mul_11_grad/tuple/control_dependency_12DeepHit/gradients/DeepHit/Sum_7_grad/Reshape/shape*
T0*
Tshape0
\
*DeepHit/gradients/DeepHit/Sum_7_grad/ShapeShapeDeepHit/Mean_6*
T0*
out_type0
¶
)DeepHit/gradients/DeepHit/Sum_7_grad/TileTile,DeepHit/gradients/DeepHit/Sum_7_grad/Reshape*DeepHit/gradients/DeepHit/Sum_7_grad/Shape*
T0*

Tmultiples0
И
SDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      
¶
MDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/ReshapeReshape`DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/control_dependency_1SDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Reshape/shape*
T0*
Tshape0
А
KDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/ConstConst*
dtype0*
valueB"`   
   
Й
JDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/TileTileMDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/ReshapeKDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Const*
T0*

Tmultiples0
К
UDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      
ђ
ODeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/ReshapeReshapebDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/control_dependency_1UDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Reshape/shape*
T0*
Tshape0
В
MDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/ConstConst*
dtype0*
valueB"
   
   
П
LDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/TileTileODeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/ReshapeMDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Const*
T0*

Tmultiples0
К
UDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      
ђ
ODeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/ReshapeReshapebDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/control_dependency_1UDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Reshape/shape*
T0*
Tshape0
В
MDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/ConstConst*
dtype0*
valueB"
   
   
П
LDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/TileTileODeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/ReshapeMDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Const*
T0*

Tmultiples0
К
UDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      
ђ
ODeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/ReshapeReshapebDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/control_dependency_1UDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Reshape/shape*
T0*
Tshape0
В
MDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/ConstConst*
dtype0*
valueB"j      
П
LDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/TileTileODeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/ReshapeMDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Const*
T0*

Tmultiples0
К
UDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      
ђ
ODeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/ReshapeReshapebDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/control_dependency_1UDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Reshape/shape*
T0*
Tshape0
В
MDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/ConstConst*
dtype0*
valueB"j      
П
LDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/TileTileODeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/ReshapeMDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Const*
T0*

Tmultiples0

JDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      
Л
DDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/ReshapeReshapeWDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/control_dependency_1JDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Reshape/shape*
T0*
Tshape0
w
BDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/ConstConst*
dtype0*
valueB"(     
о
ADeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/TileTileDDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/ReshapeBDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Const*
T0*

Tmultiples0
w
&DeepHit/gradients/DeepHit/Neg_grad/NegNeg?DeepHit/gradients/DeepHit/mul_9_grad/tuple/control_dependency_1*
T0
g
2DeepHit/gradients/DeepHit/Sum_4_grad/Reshape/shapeConst*
dtype0*
valueB"      
ƒ
,DeepHit/gradients/DeepHit/Sum_4_grad/ReshapeReshape@DeepHit/gradients/DeepHit/mul_10_grad/tuple/control_dependency_12DeepHit/gradients/DeepHit/Sum_4_grad/Reshape/shape*
T0*
Tshape0
\
*DeepHit/gradients/DeepHit/Sum_4_grad/ShapeShapeDeepHit/Mean_3*
T0*
out_type0
¶
)DeepHit/gradients/DeepHit/Sum_4_grad/TileTile,DeepHit/gradients/DeepHit/Sum_4_grad/Reshape*DeepHit/gradients/DeepHit/Sum_4_grad/Shape*
T0*

Tmultiples0
`
+DeepHit/gradients/DeepHit/Mean_6_grad/ShapeShapeDeepHit/Reshape_9*
T0*
out_type0
≠
1DeepHit/gradients/DeepHit/Mean_6_grad/BroadcastToBroadcastTo)DeepHit/gradients/DeepHit/Sum_7_grad/Tile+DeepHit/gradients/DeepHit/Mean_6_grad/Shape*
T0*

Tidx0
b
-DeepHit/gradients/DeepHit/Mean_6_grad/Shape_1ShapeDeepHit/Reshape_9*
T0*
out_type0
_
-DeepHit/gradients/DeepHit/Mean_6_grad/Shape_2ShapeDeepHit/Mean_6*
T0*
out_type0
Y
+DeepHit/gradients/DeepHit/Mean_6_grad/ConstConst*
dtype0*
valueB: 
і
*DeepHit/gradients/DeepHit/Mean_6_grad/ProdProd-DeepHit/gradients/DeepHit/Mean_6_grad/Shape_1+DeepHit/gradients/DeepHit/Mean_6_grad/Const*
T0*

Tidx0*
	keep_dims( 
[
-DeepHit/gradients/DeepHit/Mean_6_grad/Const_1Const*
dtype0*
valueB: 
Є
,DeepHit/gradients/DeepHit/Mean_6_grad/Prod_1Prod-DeepHit/gradients/DeepHit/Mean_6_grad/Shape_2-DeepHit/gradients/DeepHit/Mean_6_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
Y
/DeepHit/gradients/DeepHit/Mean_6_grad/Maximum/yConst*
dtype0*
value	B :
†
-DeepHit/gradients/DeepHit/Mean_6_grad/MaximumMaximum,DeepHit/gradients/DeepHit/Mean_6_grad/Prod_1/DeepHit/gradients/DeepHit/Mean_6_grad/Maximum/y*
T0
Ю
.DeepHit/gradients/DeepHit/Mean_6_grad/floordivFloorDiv*DeepHit/gradients/DeepHit/Mean_6_grad/Prod-DeepHit/gradients/DeepHit/Mean_6_grad/Maximum*
T0
К
*DeepHit/gradients/DeepHit/Mean_6_grad/CastCast.DeepHit/gradients/DeepHit/Mean_6_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
†
-DeepHit/gradients/DeepHit/Mean_6_grad/truedivRealDiv1DeepHit/gradients/DeepHit/Mean_6_grad/BroadcastTo*DeepHit/gradients/DeepHit/Mean_6_grad/Cast*
T0
»
NDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/ConstConstK^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Tile*
dtype0*
valueB
 *   @
“
LDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/MulMul$DeepHit/fully_connected/weights/readNDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Const*
T0
ш
NDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Mul_1MulJDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/TileLDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Mul*
T0
ћ
PDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/ConstConstM^DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Tile*
dtype0*
valueB
 *   @
Ў
NDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/MulMul&DeepHit/fully_connected_1/weights/readPDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Const*
T0
ю
PDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Mul_1MulLDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/TileNDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Mul*
T0
ћ
PDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/ConstConstM^DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Tile*
dtype0*
valueB
 *   @
Ў
NDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/MulMul&DeepHit/fully_connected_2/weights/readPDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Const*
T0
ю
PDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Mul_1MulLDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/TileNDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Mul*
T0
ћ
PDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/ConstConstM^DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Tile*
dtype0*
valueB
 *   @
Ў
NDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/MulMul&DeepHit/fully_connected_3/weights/readPDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Const*
T0
ю
PDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Mul_1MulLDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/TileNDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Mul*
T0
ћ
PDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/ConstConstM^DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Tile*
dtype0*
valueB
 *   @
Ў
NDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/MulMul&DeepHit/fully_connected_4/weights/readPDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Const*
T0
ю
PDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Mul_1MulLDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/TileNDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Mul*
T0
o
ADeepHit/gradients/DeepHit/Output/kernel/Regularizer/Abs_grad/SignSignDeepHit/Output/weights/read*
T0
÷
@DeepHit/gradients/DeepHit/Output/kernel/Regularizer/Abs_grad/mulMulADeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/TileADeepHit/gradients/DeepHit/Output/kernel/Regularizer/Abs_grad/Sign*
T0
f
1DeepHit/gradients/DeepHit/Mean_grad/Reshape/shapeConst*
dtype0*
valueB"      
®
+DeepHit/gradients/DeepHit/Mean_grad/ReshapeReshape&DeepHit/gradients/DeepHit/Neg_grad/Neg1DeepHit/gradients/DeepHit/Mean_grad/Reshape/shape*
T0*
Tshape0
Z
)DeepHit/gradients/DeepHit/Mean_grad/ShapeShapeDeepHit/add_2*
T0*
out_type0
£
(DeepHit/gradients/DeepHit/Mean_grad/TileTile+DeepHit/gradients/DeepHit/Mean_grad/Reshape)DeepHit/gradients/DeepHit/Mean_grad/Shape*
T0*

Tmultiples0
\
+DeepHit/gradients/DeepHit/Mean_grad/Shape_1ShapeDeepHit/add_2*
T0*
out_type0
T
+DeepHit/gradients/DeepHit/Mean_grad/Shape_2Const*
dtype0*
valueB 
W
)DeepHit/gradients/DeepHit/Mean_grad/ConstConst*
dtype0*
valueB: 
Ѓ
(DeepHit/gradients/DeepHit/Mean_grad/ProdProd+DeepHit/gradients/DeepHit/Mean_grad/Shape_1)DeepHit/gradients/DeepHit/Mean_grad/Const*
T0*

Tidx0*
	keep_dims( 
Y
+DeepHit/gradients/DeepHit/Mean_grad/Const_1Const*
dtype0*
valueB: 
≤
*DeepHit/gradients/DeepHit/Mean_grad/Prod_1Prod+DeepHit/gradients/DeepHit/Mean_grad/Shape_2+DeepHit/gradients/DeepHit/Mean_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
W
-DeepHit/gradients/DeepHit/Mean_grad/Maximum/yConst*
dtype0*
value	B :
Ъ
+DeepHit/gradients/DeepHit/Mean_grad/MaximumMaximum*DeepHit/gradients/DeepHit/Mean_grad/Prod_1-DeepHit/gradients/DeepHit/Mean_grad/Maximum/y*
T0
Ш
,DeepHit/gradients/DeepHit/Mean_grad/floordivFloorDiv(DeepHit/gradients/DeepHit/Mean_grad/Prod+DeepHit/gradients/DeepHit/Mean_grad/Maximum*
T0
Ж
(DeepHit/gradients/DeepHit/Mean_grad/CastCast,DeepHit/gradients/DeepHit/Mean_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
У
+DeepHit/gradients/DeepHit/Mean_grad/truedivRealDiv(DeepHit/gradients/DeepHit/Mean_grad/Tile(DeepHit/gradients/DeepHit/Mean_grad/Cast*
T0
`
+DeepHit/gradients/DeepHit/Mean_3_grad/ShapeShapeDeepHit/Reshape_6*
T0*
out_type0
≠
1DeepHit/gradients/DeepHit/Mean_3_grad/BroadcastToBroadcastTo)DeepHit/gradients/DeepHit/Sum_4_grad/Tile+DeepHit/gradients/DeepHit/Mean_3_grad/Shape*
T0*

Tidx0
b
-DeepHit/gradients/DeepHit/Mean_3_grad/Shape_1ShapeDeepHit/Reshape_6*
T0*
out_type0
_
-DeepHit/gradients/DeepHit/Mean_3_grad/Shape_2ShapeDeepHit/Mean_3*
T0*
out_type0
Y
+DeepHit/gradients/DeepHit/Mean_3_grad/ConstConst*
dtype0*
valueB: 
і
*DeepHit/gradients/DeepHit/Mean_3_grad/ProdProd-DeepHit/gradients/DeepHit/Mean_3_grad/Shape_1+DeepHit/gradients/DeepHit/Mean_3_grad/Const*
T0*

Tidx0*
	keep_dims( 
[
-DeepHit/gradients/DeepHit/Mean_3_grad/Const_1Const*
dtype0*
valueB: 
Є
,DeepHit/gradients/DeepHit/Mean_3_grad/Prod_1Prod-DeepHit/gradients/DeepHit/Mean_3_grad/Shape_2-DeepHit/gradients/DeepHit/Mean_3_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
Y
/DeepHit/gradients/DeepHit/Mean_3_grad/Maximum/yConst*
dtype0*
value	B :
†
-DeepHit/gradients/DeepHit/Mean_3_grad/MaximumMaximum,DeepHit/gradients/DeepHit/Mean_3_grad/Prod_1/DeepHit/gradients/DeepHit/Mean_3_grad/Maximum/y*
T0
Ю
.DeepHit/gradients/DeepHit/Mean_3_grad/floordivFloorDiv*DeepHit/gradients/DeepHit/Mean_3_grad/Prod-DeepHit/gradients/DeepHit/Mean_3_grad/Maximum*
T0
К
*DeepHit/gradients/DeepHit/Mean_3_grad/CastCast.DeepHit/gradients/DeepHit/Mean_3_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
†
-DeepHit/gradients/DeepHit/Mean_3_grad/truedivRealDiv1DeepHit/gradients/DeepHit/Mean_3_grad/BroadcastTo*DeepHit/gradients/DeepHit/Mean_3_grad/Cast*
T0
a
.DeepHit/gradients/DeepHit/Reshape_9_grad/ShapeShapeDeepHit/stack_2*
T0*
out_type0
±
0DeepHit/gradients/DeepHit/Reshape_9_grad/ReshapeReshape-DeepHit/gradients/DeepHit/Mean_6_grad/truediv.DeepHit/gradients/DeepHit/Reshape_9_grad/Shape*
T0*
Tshape0
[
*DeepHit/gradients/DeepHit/add_2_grad/ShapeShapeDeepHit/mul_1*
T0*
out_type0
]
,DeepHit/gradients/DeepHit/add_2_grad/Shape_1ShapeDeepHit/mul_4*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/add_2_grad/Shape,DeepHit/gradients/DeepHit/add_2_grad/Shape_1*
T0
Њ
(DeepHit/gradients/DeepHit/add_2_grad/SumSum+DeepHit/gradients/DeepHit/Mean_grad/truediv:DeepHit/gradients/DeepHit/add_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/add_2_grad/ReshapeReshape(DeepHit/gradients/DeepHit/add_2_grad/Sum*DeepHit/gradients/DeepHit/add_2_grad/Shape*
T0*
Tshape0
¬
*DeepHit/gradients/DeepHit/add_2_grad/Sum_1Sum+DeepHit/gradients/DeepHit/Mean_grad/truediv<DeepHit/gradients/DeepHit/add_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/add_2_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/add_2_grad/Sum_1,DeepHit/gradients/DeepHit/add_2_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/add_2_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/add_2_grad/Reshape/^DeepHit/gradients/DeepHit/add_2_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/add_2_grad/Reshape6^DeepHit/gradients/DeepHit/add_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/add_2_grad/Reshape
€
?DeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/add_2_grad/Reshape_16^DeepHit/gradients/DeepHit/add_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/add_2_grad/Reshape_1
a
.DeepHit/gradients/DeepHit/Reshape_6_grad/ShapeShapeDeepHit/stack_1*
T0*
out_type0
±
0DeepHit/gradients/DeepHit/Reshape_6_grad/ReshapeReshape-DeepHit/gradients/DeepHit/Mean_3_grad/truediv.DeepHit/gradients/DeepHit/Reshape_6_grad/Shape*
T0*
Tshape0
К
.DeepHit/gradients/DeepHit/stack_2_grad/unstackUnpack0DeepHit/gradients/DeepHit/Reshape_9_grad/Reshape*
T0*

axis*	
num
p
7DeepHit/gradients/DeepHit/stack_2_grad/tuple/group_depsNoOp/^DeepHit/gradients/DeepHit/stack_2_grad/unstack
Б
?DeepHit/gradients/DeepHit/stack_2_grad/tuple/control_dependencyIdentity.DeepHit/gradients/DeepHit/stack_2_grad/unstack8^DeepHit/gradients/DeepHit/stack_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/stack_2_grad/unstack
Е
ADeepHit/gradients/DeepHit/stack_2_grad/tuple/control_dependency_1Identity0DeepHit/gradients/DeepHit/stack_2_grad/unstack:18^DeepHit/gradients/DeepHit/stack_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/stack_2_grad/unstack
Z
*DeepHit/gradients/DeepHit/mul_1_grad/ShapeShapeDeepHit/Sign*
T0*
out_type0
[
,DeepHit/gradients/DeepHit/mul_1_grad/Shape_1ShapeDeepHit/Log*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/mul_1_grad/Shape,DeepHit/gradients/DeepHit/mul_1_grad/Shape_1*
T0
Д
(DeepHit/gradients/DeepHit/mul_1_grad/MulMul=DeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependencyDeepHit/Log*
T0
ї
(DeepHit/gradients/DeepHit/mul_1_grad/SumSum(DeepHit/gradients/DeepHit/mul_1_grad/Mul:DeepHit/gradients/DeepHit/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/mul_1_grad/ReshapeReshape(DeepHit/gradients/DeepHit/mul_1_grad/Sum*DeepHit/gradients/DeepHit/mul_1_grad/Shape*
T0*
Tshape0
З
*DeepHit/gradients/DeepHit/mul_1_grad/Mul_1MulDeepHit/Sign=DeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependency*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_1_grad/Sum_1Sum*DeepHit/gradients/DeepHit/mul_1_grad/Mul_1<DeepHit/gradients/DeepHit/mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_1_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/mul_1_grad/Sum_1,DeepHit/gradients/DeepHit/mul_1_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/mul_1_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/mul_1_grad/Reshape/^DeepHit/gradients/DeepHit/mul_1_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/mul_1_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/mul_1_grad/Reshape6^DeepHit/gradients/DeepHit/mul_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_1_grad/Reshape
€
?DeepHit/gradients/DeepHit/mul_1_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/mul_1_grad/Reshape_16^DeepHit/gradients/DeepHit/mul_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_1_grad/Reshape_1
]
*DeepHit/gradients/DeepHit/mul_4_grad/ShapeShapeDeepHit/mul_4/x*
T0*
out_type0
]
,DeepHit/gradients/DeepHit/mul_4_grad/Shape_1ShapeDeepHit/mul_3*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/mul_4_grad/Shape,DeepHit/gradients/DeepHit/mul_4_grad/Shape_1*
T0
И
(DeepHit/gradients/DeepHit/mul_4_grad/MulMul?DeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependency_1DeepHit/mul_3*
T0
ї
(DeepHit/gradients/DeepHit/mul_4_grad/SumSum(DeepHit/gradients/DeepHit/mul_4_grad/Mul:DeepHit/gradients/DeepHit/mul_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/mul_4_grad/ReshapeReshape(DeepHit/gradients/DeepHit/mul_4_grad/Sum*DeepHit/gradients/DeepHit/mul_4_grad/Shape*
T0*
Tshape0
М
*DeepHit/gradients/DeepHit/mul_4_grad/Mul_1MulDeepHit/mul_4/x?DeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependency_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_4_grad/Sum_1Sum*DeepHit/gradients/DeepHit/mul_4_grad/Mul_1<DeepHit/gradients/DeepHit/mul_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_4_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/mul_4_grad/Sum_1,DeepHit/gradients/DeepHit/mul_4_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/mul_4_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/mul_4_grad/Reshape/^DeepHit/gradients/DeepHit/mul_4_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/mul_4_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/mul_4_grad/Reshape6^DeepHit/gradients/DeepHit/mul_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_4_grad/Reshape
€
?DeepHit/gradients/DeepHit/mul_4_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/mul_4_grad/Reshape_16^DeepHit/gradients/DeepHit/mul_4_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_4_grad/Reshape_1
К
.DeepHit/gradients/DeepHit/stack_1_grad/unstackUnpack0DeepHit/gradients/DeepHit/Reshape_6_grad/Reshape*
T0*

axis*	
num
p
7DeepHit/gradients/DeepHit/stack_1_grad/tuple/group_depsNoOp/^DeepHit/gradients/DeepHit/stack_1_grad/unstack
Б
?DeepHit/gradients/DeepHit/stack_1_grad/tuple/control_dependencyIdentity.DeepHit/gradients/DeepHit/stack_1_grad/unstack8^DeepHit/gradients/DeepHit/stack_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/stack_1_grad/unstack
Е
ADeepHit/gradients/DeepHit/stack_1_grad/tuple/control_dependency_1Identity0DeepHit/gradients/DeepHit/stack_1_grad/unstack:18^DeepHit/gradients/DeepHit/stack_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/stack_1_grad/unstack
Z
+DeepHit/gradients/DeepHit/Mean_4_grad/ShapeShapeDeepHit/pow*
T0*
out_type0
√
1DeepHit/gradients/DeepHit/Mean_4_grad/BroadcastToBroadcastTo?DeepHit/gradients/DeepHit/stack_2_grad/tuple/control_dependency+DeepHit/gradients/DeepHit/Mean_4_grad/Shape*
T0*

Tidx0
\
-DeepHit/gradients/DeepHit/Mean_4_grad/Shape_1ShapeDeepHit/pow*
T0*
out_type0
_
-DeepHit/gradients/DeepHit/Mean_4_grad/Shape_2ShapeDeepHit/Mean_4*
T0*
out_type0
Y
+DeepHit/gradients/DeepHit/Mean_4_grad/ConstConst*
dtype0*
valueB: 
і
*DeepHit/gradients/DeepHit/Mean_4_grad/ProdProd-DeepHit/gradients/DeepHit/Mean_4_grad/Shape_1+DeepHit/gradients/DeepHit/Mean_4_grad/Const*
T0*

Tidx0*
	keep_dims( 
[
-DeepHit/gradients/DeepHit/Mean_4_grad/Const_1Const*
dtype0*
valueB: 
Є
,DeepHit/gradients/DeepHit/Mean_4_grad/Prod_1Prod-DeepHit/gradients/DeepHit/Mean_4_grad/Shape_2-DeepHit/gradients/DeepHit/Mean_4_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
Y
/DeepHit/gradients/DeepHit/Mean_4_grad/Maximum/yConst*
dtype0*
value	B :
†
-DeepHit/gradients/DeepHit/Mean_4_grad/MaximumMaximum,DeepHit/gradients/DeepHit/Mean_4_grad/Prod_1/DeepHit/gradients/DeepHit/Mean_4_grad/Maximum/y*
T0
Ю
.DeepHit/gradients/DeepHit/Mean_4_grad/floordivFloorDiv*DeepHit/gradients/DeepHit/Mean_4_grad/Prod-DeepHit/gradients/DeepHit/Mean_4_grad/Maximum*
T0
К
*DeepHit/gradients/DeepHit/Mean_4_grad/CastCast.DeepHit/gradients/DeepHit/Mean_4_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
†
-DeepHit/gradients/DeepHit/Mean_4_grad/truedivRealDiv1DeepHit/gradients/DeepHit/Mean_4_grad/BroadcastTo*DeepHit/gradients/DeepHit/Mean_4_grad/Cast*
T0
\
+DeepHit/gradients/DeepHit/Mean_5_grad/ShapeShapeDeepHit/pow_1*
T0*
out_type0
≈
1DeepHit/gradients/DeepHit/Mean_5_grad/BroadcastToBroadcastToADeepHit/gradients/DeepHit/stack_2_grad/tuple/control_dependency_1+DeepHit/gradients/DeepHit/Mean_5_grad/Shape*
T0*

Tidx0
^
-DeepHit/gradients/DeepHit/Mean_5_grad/Shape_1ShapeDeepHit/pow_1*
T0*
out_type0
_
-DeepHit/gradients/DeepHit/Mean_5_grad/Shape_2ShapeDeepHit/Mean_5*
T0*
out_type0
Y
+DeepHit/gradients/DeepHit/Mean_5_grad/ConstConst*
dtype0*
valueB: 
і
*DeepHit/gradients/DeepHit/Mean_5_grad/ProdProd-DeepHit/gradients/DeepHit/Mean_5_grad/Shape_1+DeepHit/gradients/DeepHit/Mean_5_grad/Const*
T0*

Tidx0*
	keep_dims( 
[
-DeepHit/gradients/DeepHit/Mean_5_grad/Const_1Const*
dtype0*
valueB: 
Є
,DeepHit/gradients/DeepHit/Mean_5_grad/Prod_1Prod-DeepHit/gradients/DeepHit/Mean_5_grad/Shape_2-DeepHit/gradients/DeepHit/Mean_5_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
Y
/DeepHit/gradients/DeepHit/Mean_5_grad/Maximum/yConst*
dtype0*
value	B :
†
-DeepHit/gradients/DeepHit/Mean_5_grad/MaximumMaximum,DeepHit/gradients/DeepHit/Mean_5_grad/Prod_1/DeepHit/gradients/DeepHit/Mean_5_grad/Maximum/y*
T0
Ю
.DeepHit/gradients/DeepHit/Mean_5_grad/floordivFloorDiv*DeepHit/gradients/DeepHit/Mean_5_grad/Prod-DeepHit/gradients/DeepHit/Mean_5_grad/Maximum*
T0
К
*DeepHit/gradients/DeepHit/Mean_5_grad/CastCast.DeepHit/gradients/DeepHit/Mean_5_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
†
-DeepHit/gradients/DeepHit/Mean_5_grad/truedivRealDiv1DeepHit/gradients/DeepHit/Mean_5_grad/BroadcastTo*DeepHit/gradients/DeepHit/Mean_5_grad/Cast*
T0
У
-DeepHit/gradients/DeepHit/Log_grad/Reciprocal
ReciprocalDeepHit/add@^DeepHit/gradients/DeepHit/mul_1_grad/tuple/control_dependency_1*
T0
¶
&DeepHit/gradients/DeepHit/Log_grad/mulMul?DeepHit/gradients/DeepHit/mul_1_grad/tuple/control_dependency_1-DeepHit/gradients/DeepHit/Log_grad/Reciprocal*
T0
[
*DeepHit/gradients/DeepHit/mul_3_grad/ShapeShapeDeepHit/sub_3*
T0*
out_type0
]
,DeepHit/gradients/DeepHit/mul_3_grad/Shape_1ShapeDeepHit/Log_1*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/mul_3_grad/Shape,DeepHit/gradients/DeepHit/mul_3_grad/Shape_1*
T0
И
(DeepHit/gradients/DeepHit/mul_3_grad/MulMul?DeepHit/gradients/DeepHit/mul_4_grad/tuple/control_dependency_1DeepHit/Log_1*
T0
ї
(DeepHit/gradients/DeepHit/mul_3_grad/SumSum(DeepHit/gradients/DeepHit/mul_3_grad/Mul:DeepHit/gradients/DeepHit/mul_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/mul_3_grad/ReshapeReshape(DeepHit/gradients/DeepHit/mul_3_grad/Sum*DeepHit/gradients/DeepHit/mul_3_grad/Shape*
T0*
Tshape0
К
*DeepHit/gradients/DeepHit/mul_3_grad/Mul_1MulDeepHit/sub_3?DeepHit/gradients/DeepHit/mul_4_grad/tuple/control_dependency_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_3_grad/Sum_1Sum*DeepHit/gradients/DeepHit/mul_3_grad/Mul_1<DeepHit/gradients/DeepHit/mul_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_3_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/mul_3_grad/Sum_1,DeepHit/gradients/DeepHit/mul_3_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/mul_3_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/mul_3_grad/Reshape/^DeepHit/gradients/DeepHit/mul_3_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/mul_3_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/mul_3_grad/Reshape6^DeepHit/gradients/DeepHit/mul_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_3_grad/Reshape
€
?DeepHit/gradients/DeepHit/mul_3_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/mul_3_grad/Reshape_16^DeepHit/gradients/DeepHit/mul_3_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_3_grad/Reshape_1
\
+DeepHit/gradients/DeepHit/Mean_1_grad/ShapeShapeDeepHit/mul_5*
T0*
out_type0
√
1DeepHit/gradients/DeepHit/Mean_1_grad/BroadcastToBroadcastTo?DeepHit/gradients/DeepHit/stack_1_grad/tuple/control_dependency+DeepHit/gradients/DeepHit/Mean_1_grad/Shape*
T0*

Tidx0
^
-DeepHit/gradients/DeepHit/Mean_1_grad/Shape_1ShapeDeepHit/mul_5*
T0*
out_type0
_
-DeepHit/gradients/DeepHit/Mean_1_grad/Shape_2ShapeDeepHit/Mean_1*
T0*
out_type0
Y
+DeepHit/gradients/DeepHit/Mean_1_grad/ConstConst*
dtype0*
valueB: 
і
*DeepHit/gradients/DeepHit/Mean_1_grad/ProdProd-DeepHit/gradients/DeepHit/Mean_1_grad/Shape_1+DeepHit/gradients/DeepHit/Mean_1_grad/Const*
T0*

Tidx0*
	keep_dims( 
[
-DeepHit/gradients/DeepHit/Mean_1_grad/Const_1Const*
dtype0*
valueB: 
Є
,DeepHit/gradients/DeepHit/Mean_1_grad/Prod_1Prod-DeepHit/gradients/DeepHit/Mean_1_grad/Shape_2-DeepHit/gradients/DeepHit/Mean_1_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
Y
/DeepHit/gradients/DeepHit/Mean_1_grad/Maximum/yConst*
dtype0*
value	B :
†
-DeepHit/gradients/DeepHit/Mean_1_grad/MaximumMaximum,DeepHit/gradients/DeepHit/Mean_1_grad/Prod_1/DeepHit/gradients/DeepHit/Mean_1_grad/Maximum/y*
T0
Ю
.DeepHit/gradients/DeepHit/Mean_1_grad/floordivFloorDiv*DeepHit/gradients/DeepHit/Mean_1_grad/Prod-DeepHit/gradients/DeepHit/Mean_1_grad/Maximum*
T0
К
*DeepHit/gradients/DeepHit/Mean_1_grad/CastCast.DeepHit/gradients/DeepHit/Mean_1_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
†
-DeepHit/gradients/DeepHit/Mean_1_grad/truedivRealDiv1DeepHit/gradients/DeepHit/Mean_1_grad/BroadcastTo*DeepHit/gradients/DeepHit/Mean_1_grad/Cast*
T0
\
+DeepHit/gradients/DeepHit/Mean_2_grad/ShapeShapeDeepHit/mul_6*
T0*
out_type0
≈
1DeepHit/gradients/DeepHit/Mean_2_grad/BroadcastToBroadcastToADeepHit/gradients/DeepHit/stack_1_grad/tuple/control_dependency_1+DeepHit/gradients/DeepHit/Mean_2_grad/Shape*
T0*

Tidx0
^
-DeepHit/gradients/DeepHit/Mean_2_grad/Shape_1ShapeDeepHit/mul_6*
T0*
out_type0
_
-DeepHit/gradients/DeepHit/Mean_2_grad/Shape_2ShapeDeepHit/Mean_2*
T0*
out_type0
Y
+DeepHit/gradients/DeepHit/Mean_2_grad/ConstConst*
dtype0*
valueB: 
і
*DeepHit/gradients/DeepHit/Mean_2_grad/ProdProd-DeepHit/gradients/DeepHit/Mean_2_grad/Shape_1+DeepHit/gradients/DeepHit/Mean_2_grad/Const*
T0*

Tidx0*
	keep_dims( 
[
-DeepHit/gradients/DeepHit/Mean_2_grad/Const_1Const*
dtype0*
valueB: 
Є
,DeepHit/gradients/DeepHit/Mean_2_grad/Prod_1Prod-DeepHit/gradients/DeepHit/Mean_2_grad/Shape_2-DeepHit/gradients/DeepHit/Mean_2_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
Y
/DeepHit/gradients/DeepHit/Mean_2_grad/Maximum/yConst*
dtype0*
value	B :
†
-DeepHit/gradients/DeepHit/Mean_2_grad/MaximumMaximum,DeepHit/gradients/DeepHit/Mean_2_grad/Prod_1/DeepHit/gradients/DeepHit/Mean_2_grad/Maximum/y*
T0
Ю
.DeepHit/gradients/DeepHit/Mean_2_grad/floordivFloorDiv*DeepHit/gradients/DeepHit/Mean_2_grad/Prod-DeepHit/gradients/DeepHit/Mean_2_grad/Maximum*
T0
К
*DeepHit/gradients/DeepHit/Mean_2_grad/CastCast.DeepHit/gradients/DeepHit/Mean_2_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
†
-DeepHit/gradients/DeepHit/Mean_2_grad/truedivRealDiv1DeepHit/gradients/DeepHit/Mean_2_grad/BroadcastTo*DeepHit/gradients/DeepHit/Mean_2_grad/Cast*
T0
Y
(DeepHit/gradients/DeepHit/pow_grad/ShapeShapeDeepHit/sub_8*
T0*
out_type0
[
*DeepHit/gradients/DeepHit/pow_grad/Shape_1ShapeDeepHit/pow/y*
T0*
out_type0
∞
8DeepHit/gradients/DeepHit/pow_grad/BroadcastGradientArgsBroadcastGradientArgs(DeepHit/gradients/DeepHit/pow_grad/Shape*DeepHit/gradients/DeepHit/pow_grad/Shape_1*
T0
t
&DeepHit/gradients/DeepHit/pow_grad/mulMul-DeepHit/gradients/DeepHit/Mean_4_grad/truedivDeepHit/pow/y*
T0
U
(DeepHit/gradients/DeepHit/pow_grad/sub/yConst*
dtype0*
valueB
 *  А?
o
&DeepHit/gradients/DeepHit/pow_grad/subSubDeepHit/pow/y(DeepHit/gradients/DeepHit/pow_grad/sub/y*
T0
m
&DeepHit/gradients/DeepHit/pow_grad/PowPowDeepHit/sub_8&DeepHit/gradients/DeepHit/pow_grad/sub*
T0
И
(DeepHit/gradients/DeepHit/pow_grad/mul_1Mul&DeepHit/gradients/DeepHit/pow_grad/mul&DeepHit/gradients/DeepHit/pow_grad/Pow*
T0
Ј
&DeepHit/gradients/DeepHit/pow_grad/SumSum(DeepHit/gradients/DeepHit/pow_grad/mul_18DeepHit/gradients/DeepHit/pow_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
Ю
*DeepHit/gradients/DeepHit/pow_grad/ReshapeReshape&DeepHit/gradients/DeepHit/pow_grad/Sum(DeepHit/gradients/DeepHit/pow_grad/Shape*
T0*
Tshape0
Y
,DeepHit/gradients/DeepHit/pow_grad/Greater/yConst*
dtype0*
valueB
 *    
{
*DeepHit/gradients/DeepHit/pow_grad/GreaterGreaterDeepHit/sub_8,DeepHit/gradients/DeepHit/pow_grad/Greater/y*
T0
c
2DeepHit/gradients/DeepHit/pow_grad/ones_like/ShapeShapeDeepHit/sub_8*
T0*
out_type0
_
2DeepHit/gradients/DeepHit/pow_grad/ones_like/ConstConst*
dtype0*
valueB
 *  А?
Ј
,DeepHit/gradients/DeepHit/pow_grad/ones_likeFill2DeepHit/gradients/DeepHit/pow_grad/ones_like/Shape2DeepHit/gradients/DeepHit/pow_grad/ones_like/Const*
T0*

index_type0
•
)DeepHit/gradients/DeepHit/pow_grad/SelectSelect*DeepHit/gradients/DeepHit/pow_grad/GreaterDeepHit/sub_8,DeepHit/gradients/DeepHit/pow_grad/ones_like*
T0
a
&DeepHit/gradients/DeepHit/pow_grad/LogLog)DeepHit/gradients/DeepHit/pow_grad/Select*
T0
R
-DeepHit/gradients/DeepHit/pow_grad/zeros_like	ZerosLikeDeepHit/sub_8*
T0
Ѕ
+DeepHit/gradients/DeepHit/pow_grad/Select_1Select*DeepHit/gradients/DeepHit/pow_grad/Greater&DeepHit/gradients/DeepHit/pow_grad/Log-DeepHit/gradients/DeepHit/pow_grad/zeros_like*
T0
t
(DeepHit/gradients/DeepHit/pow_grad/mul_2Mul-DeepHit/gradients/DeepHit/Mean_4_grad/truedivDeepHit/pow*
T0
П
(DeepHit/gradients/DeepHit/pow_grad/mul_3Mul(DeepHit/gradients/DeepHit/pow_grad/mul_2+DeepHit/gradients/DeepHit/pow_grad/Select_1*
T0
ї
(DeepHit/gradients/DeepHit/pow_grad/Sum_1Sum(DeepHit/gradients/DeepHit/pow_grad/mul_3:DeepHit/gradients/DeepHit/pow_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/pow_grad/Reshape_1Reshape(DeepHit/gradients/DeepHit/pow_grad/Sum_1*DeepHit/gradients/DeepHit/pow_grad/Shape_1*
T0*
Tshape0
Ч
3DeepHit/gradients/DeepHit/pow_grad/tuple/group_depsNoOp+^DeepHit/gradients/DeepHit/pow_grad/Reshape-^DeepHit/gradients/DeepHit/pow_grad/Reshape_1
с
;DeepHit/gradients/DeepHit/pow_grad/tuple/control_dependencyIdentity*DeepHit/gradients/DeepHit/pow_grad/Reshape4^DeepHit/gradients/DeepHit/pow_grad/tuple/group_deps*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/pow_grad/Reshape
ч
=DeepHit/gradients/DeepHit/pow_grad/tuple/control_dependency_1Identity,DeepHit/gradients/DeepHit/pow_grad/Reshape_14^DeepHit/gradients/DeepHit/pow_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/pow_grad/Reshape_1
[
*DeepHit/gradients/DeepHit/pow_1_grad/ShapeShapeDeepHit/sub_9*
T0*
out_type0
_
,DeepHit/gradients/DeepHit/pow_1_grad/Shape_1ShapeDeepHit/pow_1/y*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/pow_1_grad/Shape,DeepHit/gradients/DeepHit/pow_1_grad/Shape_1*
T0
x
(DeepHit/gradients/DeepHit/pow_1_grad/mulMul-DeepHit/gradients/DeepHit/Mean_5_grad/truedivDeepHit/pow_1/y*
T0
W
*DeepHit/gradients/DeepHit/pow_1_grad/sub/yConst*
dtype0*
valueB
 *  А?
u
(DeepHit/gradients/DeepHit/pow_1_grad/subSubDeepHit/pow_1/y*DeepHit/gradients/DeepHit/pow_1_grad/sub/y*
T0
q
(DeepHit/gradients/DeepHit/pow_1_grad/PowPowDeepHit/sub_9(DeepHit/gradients/DeepHit/pow_1_grad/sub*
T0
О
*DeepHit/gradients/DeepHit/pow_1_grad/mul_1Mul(DeepHit/gradients/DeepHit/pow_1_grad/mul(DeepHit/gradients/DeepHit/pow_1_grad/Pow*
T0
љ
(DeepHit/gradients/DeepHit/pow_1_grad/SumSum*DeepHit/gradients/DeepHit/pow_1_grad/mul_1:DeepHit/gradients/DeepHit/pow_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/pow_1_grad/ReshapeReshape(DeepHit/gradients/DeepHit/pow_1_grad/Sum*DeepHit/gradients/DeepHit/pow_1_grad/Shape*
T0*
Tshape0
[
.DeepHit/gradients/DeepHit/pow_1_grad/Greater/yConst*
dtype0*
valueB
 *    

,DeepHit/gradients/DeepHit/pow_1_grad/GreaterGreaterDeepHit/sub_9.DeepHit/gradients/DeepHit/pow_1_grad/Greater/y*
T0
e
4DeepHit/gradients/DeepHit/pow_1_grad/ones_like/ShapeShapeDeepHit/sub_9*
T0*
out_type0
a
4DeepHit/gradients/DeepHit/pow_1_grad/ones_like/ConstConst*
dtype0*
valueB
 *  А?
љ
.DeepHit/gradients/DeepHit/pow_1_grad/ones_likeFill4DeepHit/gradients/DeepHit/pow_1_grad/ones_like/Shape4DeepHit/gradients/DeepHit/pow_1_grad/ones_like/Const*
T0*

index_type0
Ђ
+DeepHit/gradients/DeepHit/pow_1_grad/SelectSelect,DeepHit/gradients/DeepHit/pow_1_grad/GreaterDeepHit/sub_9.DeepHit/gradients/DeepHit/pow_1_grad/ones_like*
T0
e
(DeepHit/gradients/DeepHit/pow_1_grad/LogLog+DeepHit/gradients/DeepHit/pow_1_grad/Select*
T0
T
/DeepHit/gradients/DeepHit/pow_1_grad/zeros_like	ZerosLikeDeepHit/sub_9*
T0
…
-DeepHit/gradients/DeepHit/pow_1_grad/Select_1Select,DeepHit/gradients/DeepHit/pow_1_grad/Greater(DeepHit/gradients/DeepHit/pow_1_grad/Log/DeepHit/gradients/DeepHit/pow_1_grad/zeros_like*
T0
x
*DeepHit/gradients/DeepHit/pow_1_grad/mul_2Mul-DeepHit/gradients/DeepHit/Mean_5_grad/truedivDeepHit/pow_1*
T0
Х
*DeepHit/gradients/DeepHit/pow_1_grad/mul_3Mul*DeepHit/gradients/DeepHit/pow_1_grad/mul_2-DeepHit/gradients/DeepHit/pow_1_grad/Select_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/pow_1_grad/Sum_1Sum*DeepHit/gradients/DeepHit/pow_1_grad/mul_3<DeepHit/gradients/DeepHit/pow_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/pow_1_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/pow_1_grad/Sum_1,DeepHit/gradients/DeepHit/pow_1_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/pow_1_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/pow_1_grad/Reshape/^DeepHit/gradients/DeepHit/pow_1_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/pow_1_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/pow_1_grad/Reshape6^DeepHit/gradients/DeepHit/pow_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/pow_1_grad/Reshape
€
?DeepHit/gradients/DeepHit/pow_1_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/pow_1_grad/Reshape_16^DeepHit/gradients/DeepHit/pow_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/pow_1_grad/Reshape_1
Y
(DeepHit/gradients/DeepHit/add_grad/ShapeShapeDeepHit/Sum_1*
T0*
out_type0
[
*DeepHit/gradients/DeepHit/add_grad/Shape_1ShapeDeepHit/add/y*
T0*
out_type0
∞
8DeepHit/gradients/DeepHit/add_grad/BroadcastGradientArgsBroadcastGradientArgs(DeepHit/gradients/DeepHit/add_grad/Shape*DeepHit/gradients/DeepHit/add_grad/Shape_1*
T0
µ
&DeepHit/gradients/DeepHit/add_grad/SumSum&DeepHit/gradients/DeepHit/Log_grad/mul8DeepHit/gradients/DeepHit/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
Ю
*DeepHit/gradients/DeepHit/add_grad/ReshapeReshape&DeepHit/gradients/DeepHit/add_grad/Sum(DeepHit/gradients/DeepHit/add_grad/Shape*
T0*
Tshape0
є
(DeepHit/gradients/DeepHit/add_grad/Sum_1Sum&DeepHit/gradients/DeepHit/Log_grad/mul:DeepHit/gradients/DeepHit/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/add_grad/Reshape_1Reshape(DeepHit/gradients/DeepHit/add_grad/Sum_1*DeepHit/gradients/DeepHit/add_grad/Shape_1*
T0*
Tshape0
Ч
3DeepHit/gradients/DeepHit/add_grad/tuple/group_depsNoOp+^DeepHit/gradients/DeepHit/add_grad/Reshape-^DeepHit/gradients/DeepHit/add_grad/Reshape_1
с
;DeepHit/gradients/DeepHit/add_grad/tuple/control_dependencyIdentity*DeepHit/gradients/DeepHit/add_grad/Reshape4^DeepHit/gradients/DeepHit/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/add_grad/Reshape
ч
=DeepHit/gradients/DeepHit/add_grad/tuple/control_dependency_1Identity,DeepHit/gradients/DeepHit/add_grad/Reshape_14^DeepHit/gradients/DeepHit/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/add_grad/Reshape_1
Ч
/DeepHit/gradients/DeepHit/Log_1_grad/Reciprocal
ReciprocalDeepHit/add_1@^DeepHit/gradients/DeepHit/mul_3_grad/tuple/control_dependency_1*
T0
™
(DeepHit/gradients/DeepHit/Log_1_grad/mulMul?DeepHit/gradients/DeepHit/mul_3_grad/tuple/control_dependency_1/DeepHit/gradients/DeepHit/Log_1_grad/Reciprocal*
T0
^
*DeepHit/gradients/DeepHit/mul_5_grad/ShapeShapeDeepHit/MatMul_4*
T0*
out_type0
[
,DeepHit/gradients/DeepHit/mul_5_grad/Shape_1ShapeDeepHit/Exp*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/mul_5_grad/Shape,DeepHit/gradients/DeepHit/mul_5_grad/Shape_1*
T0
t
(DeepHit/gradients/DeepHit/mul_5_grad/MulMul-DeepHit/gradients/DeepHit/Mean_1_grad/truedivDeepHit/Exp*
T0
ї
(DeepHit/gradients/DeepHit/mul_5_grad/SumSum(DeepHit/gradients/DeepHit/mul_5_grad/Mul:DeepHit/gradients/DeepHit/mul_5_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/mul_5_grad/ReshapeReshape(DeepHit/gradients/DeepHit/mul_5_grad/Sum*DeepHit/gradients/DeepHit/mul_5_grad/Shape*
T0*
Tshape0
{
*DeepHit/gradients/DeepHit/mul_5_grad/Mul_1MulDeepHit/MatMul_4-DeepHit/gradients/DeepHit/Mean_1_grad/truediv*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_5_grad/Sum_1Sum*DeepHit/gradients/DeepHit/mul_5_grad/Mul_1<DeepHit/gradients/DeepHit/mul_5_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_5_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/mul_5_grad/Sum_1,DeepHit/gradients/DeepHit/mul_5_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/mul_5_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/mul_5_grad/Reshape/^DeepHit/gradients/DeepHit/mul_5_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/mul_5_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/mul_5_grad/Reshape6^DeepHit/gradients/DeepHit/mul_5_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_5_grad/Reshape
€
?DeepHit/gradients/DeepHit/mul_5_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/mul_5_grad/Reshape_16^DeepHit/gradients/DeepHit/mul_5_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_5_grad/Reshape_1
^
*DeepHit/gradients/DeepHit/mul_6_grad/ShapeShapeDeepHit/MatMul_9*
T0*
out_type0
]
,DeepHit/gradients/DeepHit/mul_6_grad/Shape_1ShapeDeepHit/Exp_1*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/mul_6_grad/Shape,DeepHit/gradients/DeepHit/mul_6_grad/Shape_1*
T0
v
(DeepHit/gradients/DeepHit/mul_6_grad/MulMul-DeepHit/gradients/DeepHit/Mean_2_grad/truedivDeepHit/Exp_1*
T0
ї
(DeepHit/gradients/DeepHit/mul_6_grad/SumSum(DeepHit/gradients/DeepHit/mul_6_grad/Mul:DeepHit/gradients/DeepHit/mul_6_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/mul_6_grad/ReshapeReshape(DeepHit/gradients/DeepHit/mul_6_grad/Sum*DeepHit/gradients/DeepHit/mul_6_grad/Shape*
T0*
Tshape0
{
*DeepHit/gradients/DeepHit/mul_6_grad/Mul_1MulDeepHit/MatMul_9-DeepHit/gradients/DeepHit/Mean_2_grad/truediv*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_6_grad/Sum_1Sum*DeepHit/gradients/DeepHit/mul_6_grad/Mul_1<DeepHit/gradients/DeepHit/mul_6_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_6_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/mul_6_grad/Sum_1,DeepHit/gradients/DeepHit/mul_6_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/mul_6_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/mul_6_grad/Reshape/^DeepHit/gradients/DeepHit/mul_6_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/mul_6_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/mul_6_grad/Reshape6^DeepHit/gradients/DeepHit/mul_6_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_6_grad/Reshape
€
?DeepHit/gradients/DeepHit/mul_6_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/mul_6_grad/Reshape_16^DeepHit/gradients/DeepHit/mul_6_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_6_grad/Reshape_1
[
*DeepHit/gradients/DeepHit/sub_8_grad/ShapeShapeDeepHit/Sum_5*
T0*
out_type0
^
,DeepHit/gradients/DeepHit/sub_8_grad/Shape_1ShapeDeepHit/Cast_2*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/sub_8_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/sub_8_grad/Shape,DeepHit/gradients/DeepHit/sub_8_grad/Shape_1*
T0
ќ
(DeepHit/gradients/DeepHit/sub_8_grad/SumSum;DeepHit/gradients/DeepHit/pow_grad/tuple/control_dependency:DeepHit/gradients/DeepHit/sub_8_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/sub_8_grad/ReshapeReshape(DeepHit/gradients/DeepHit/sub_8_grad/Sum*DeepHit/gradients/DeepHit/sub_8_grad/Shape*
T0*
Tshape0
u
(DeepHit/gradients/DeepHit/sub_8_grad/NegNeg;DeepHit/gradients/DeepHit/pow_grad/tuple/control_dependency*
T0
њ
*DeepHit/gradients/DeepHit/sub_8_grad/Sum_1Sum(DeepHit/gradients/DeepHit/sub_8_grad/Neg<DeepHit/gradients/DeepHit/sub_8_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/sub_8_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/sub_8_grad/Sum_1,DeepHit/gradients/DeepHit/sub_8_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/sub_8_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/sub_8_grad/Reshape/^DeepHit/gradients/DeepHit/sub_8_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/sub_8_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/sub_8_grad/Reshape6^DeepHit/gradients/DeepHit/sub_8_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/sub_8_grad/Reshape
€
?DeepHit/gradients/DeepHit/sub_8_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/sub_8_grad/Reshape_16^DeepHit/gradients/DeepHit/sub_8_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/sub_8_grad/Reshape_1
[
*DeepHit/gradients/DeepHit/sub_9_grad/ShapeShapeDeepHit/Sum_6*
T0*
out_type0
^
,DeepHit/gradients/DeepHit/sub_9_grad/Shape_1ShapeDeepHit/Cast_3*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/sub_9_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/sub_9_grad/Shape,DeepHit/gradients/DeepHit/sub_9_grad/Shape_1*
T0
–
(DeepHit/gradients/DeepHit/sub_9_grad/SumSum=DeepHit/gradients/DeepHit/pow_1_grad/tuple/control_dependency:DeepHit/gradients/DeepHit/sub_9_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/sub_9_grad/ReshapeReshape(DeepHit/gradients/DeepHit/sub_9_grad/Sum*DeepHit/gradients/DeepHit/sub_9_grad/Shape*
T0*
Tshape0
w
(DeepHit/gradients/DeepHit/sub_9_grad/NegNeg=DeepHit/gradients/DeepHit/pow_1_grad/tuple/control_dependency*
T0
њ
*DeepHit/gradients/DeepHit/sub_9_grad/Sum_1Sum(DeepHit/gradients/DeepHit/sub_9_grad/Neg<DeepHit/gradients/DeepHit/sub_9_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/sub_9_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/sub_9_grad/Sum_1,DeepHit/gradients/DeepHit/sub_9_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/sub_9_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/sub_9_grad/Reshape/^DeepHit/gradients/DeepHit/sub_9_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/sub_9_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/sub_9_grad/Reshape6^DeepHit/gradients/DeepHit/sub_9_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/sub_9_grad/Reshape
€
?DeepHit/gradients/DeepHit/sub_9_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/sub_9_grad/Reshape_16^DeepHit/gradients/DeepHit/sub_9_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/sub_9_grad/Reshape_1
Y
*DeepHit/gradients/DeepHit/Sum_1_grad/ShapeShapeDeepHit/Sum*
T0*
out_type0
љ
0DeepHit/gradients/DeepHit/Sum_1_grad/BroadcastToBroadcastTo;DeepHit/gradients/DeepHit/add_grad/tuple/control_dependency*DeepHit/gradients/DeepHit/Sum_1_grad/Shape*
T0*

Tidx0
[
*DeepHit/gradients/DeepHit/add_1_grad/ShapeShapeDeepHit/Sum_3*
T0*
out_type0
_
,DeepHit/gradients/DeepHit/add_1_grad/Shape_1ShapeDeepHit/add_1/y*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/add_1_grad/Shape,DeepHit/gradients/DeepHit/add_1_grad/Shape_1*
T0
ї
(DeepHit/gradients/DeepHit/add_1_grad/SumSum(DeepHit/gradients/DeepHit/Log_1_grad/mul:DeepHit/gradients/DeepHit/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/add_1_grad/ReshapeReshape(DeepHit/gradients/DeepHit/add_1_grad/Sum*DeepHit/gradients/DeepHit/add_1_grad/Shape*
T0*
Tshape0
њ
*DeepHit/gradients/DeepHit/add_1_grad/Sum_1Sum(DeepHit/gradients/DeepHit/Log_1_grad/mul<DeepHit/gradients/DeepHit/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/add_1_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/add_1_grad/Sum_1,DeepHit/gradients/DeepHit/add_1_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/add_1_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/add_1_grad/Reshape/^DeepHit/gradients/DeepHit/add_1_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/add_1_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/add_1_grad/Reshape6^DeepHit/gradients/DeepHit/add_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/add_1_grad/Reshape
€
?DeepHit/gradients/DeepHit/add_1_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/add_1_grad/Reshape_16^DeepHit/gradients/DeepHit/add_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/add_1_grad/Reshape_1
Д
&DeepHit/gradients/DeepHit/Exp_grad/mulMul?DeepHit/gradients/DeepHit/mul_5_grad/tuple/control_dependency_1DeepHit/Exp*
T0
И
(DeepHit/gradients/DeepHit/Exp_1_grad/mulMul?DeepHit/gradients/DeepHit/mul_6_grad/tuple/control_dependency_1DeepHit/Exp_1*
T0
[
*DeepHit/gradients/DeepHit/Sum_5_grad/ShapeShapeDeepHit/mul_7*
T0*
out_type0
Т
)DeepHit/gradients/DeepHit/Sum_5_grad/SizeConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape*
dtype0*
value	B :
≈
(DeepHit/gradients/DeepHit/Sum_5_grad/addAddV2DeepHit/Sum_5/reduction_indices)DeepHit/gradients/DeepHit/Sum_5_grad/Size*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape
—
(DeepHit/gradients/DeepHit/Sum_5_grad/modFloorMod(DeepHit/gradients/DeepHit/Sum_5_grad/add)DeepHit/gradients/DeepHit/Sum_5_grad/Size*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape
Ф
,DeepHit/gradients/DeepHit/Sum_5_grad/Shape_1Const*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape*
dtype0*
valueB 
Щ
0DeepHit/gradients/DeepHit/Sum_5_grad/range/startConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape*
dtype0*
value	B : 
Щ
0DeepHit/gradients/DeepHit/Sum_5_grad/range/deltaConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape*
dtype0*
value	B :
Н
*DeepHit/gradients/DeepHit/Sum_5_grad/rangeRange0DeepHit/gradients/DeepHit/Sum_5_grad/range/start)DeepHit/gradients/DeepHit/Sum_5_grad/Size0DeepHit/gradients/DeepHit/Sum_5_grad/range/delta*

Tidx0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape
Ш
/DeepHit/gradients/DeepHit/Sum_5_grad/ones/ConstConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape*
dtype0*
value	B :
к
)DeepHit/gradients/DeepHit/Sum_5_grad/onesFill,DeepHit/gradients/DeepHit/Sum_5_grad/Shape_1/DeepHit/gradients/DeepHit/Sum_5_grad/ones/Const*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape*

index_type0
Ѕ
2DeepHit/gradients/DeepHit/Sum_5_grad/DynamicStitchDynamicStitch*DeepHit/gradients/DeepHit/Sum_5_grad/range(DeepHit/gradients/DeepHit/Sum_5_grad/mod*DeepHit/gradients/DeepHit/Sum_5_grad/Shape)DeepHit/gradients/DeepHit/Sum_5_grad/ones*
N*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape
Ѕ
,DeepHit/gradients/DeepHit/Sum_5_grad/ReshapeReshape=DeepHit/gradients/DeepHit/sub_8_grad/tuple/control_dependency2DeepHit/gradients/DeepHit/Sum_5_grad/DynamicStitch*
T0*
Tshape0
Ѓ
0DeepHit/gradients/DeepHit/Sum_5_grad/BroadcastToBroadcastTo,DeepHit/gradients/DeepHit/Sum_5_grad/Reshape*DeepHit/gradients/DeepHit/Sum_5_grad/Shape*
T0*

Tidx0
[
*DeepHit/gradients/DeepHit/Sum_6_grad/ShapeShapeDeepHit/mul_8*
T0*
out_type0
Т
)DeepHit/gradients/DeepHit/Sum_6_grad/SizeConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape*
dtype0*
value	B :
≈
(DeepHit/gradients/DeepHit/Sum_6_grad/addAddV2DeepHit/Sum_6/reduction_indices)DeepHit/gradients/DeepHit/Sum_6_grad/Size*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape
—
(DeepHit/gradients/DeepHit/Sum_6_grad/modFloorMod(DeepHit/gradients/DeepHit/Sum_6_grad/add)DeepHit/gradients/DeepHit/Sum_6_grad/Size*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape
Ф
,DeepHit/gradients/DeepHit/Sum_6_grad/Shape_1Const*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape*
dtype0*
valueB 
Щ
0DeepHit/gradients/DeepHit/Sum_6_grad/range/startConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape*
dtype0*
value	B : 
Щ
0DeepHit/gradients/DeepHit/Sum_6_grad/range/deltaConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape*
dtype0*
value	B :
Н
*DeepHit/gradients/DeepHit/Sum_6_grad/rangeRange0DeepHit/gradients/DeepHit/Sum_6_grad/range/start)DeepHit/gradients/DeepHit/Sum_6_grad/Size0DeepHit/gradients/DeepHit/Sum_6_grad/range/delta*

Tidx0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape
Ш
/DeepHit/gradients/DeepHit/Sum_6_grad/ones/ConstConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape*
dtype0*
value	B :
к
)DeepHit/gradients/DeepHit/Sum_6_grad/onesFill,DeepHit/gradients/DeepHit/Sum_6_grad/Shape_1/DeepHit/gradients/DeepHit/Sum_6_grad/ones/Const*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape*

index_type0
Ѕ
2DeepHit/gradients/DeepHit/Sum_6_grad/DynamicStitchDynamicStitch*DeepHit/gradients/DeepHit/Sum_6_grad/range(DeepHit/gradients/DeepHit/Sum_6_grad/mod*DeepHit/gradients/DeepHit/Sum_6_grad/Shape)DeepHit/gradients/DeepHit/Sum_6_grad/ones*
N*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape
Ѕ
,DeepHit/gradients/DeepHit/Sum_6_grad/ReshapeReshape=DeepHit/gradients/DeepHit/sub_9_grad/tuple/control_dependency2DeepHit/gradients/DeepHit/Sum_6_grad/DynamicStitch*
T0*
Tshape0
Ѓ
0DeepHit/gradients/DeepHit/Sum_6_grad/BroadcastToBroadcastTo,DeepHit/gradients/DeepHit/Sum_6_grad/Reshape*DeepHit/gradients/DeepHit/Sum_6_grad/Shape*
T0*

Tidx0
W
(DeepHit/gradients/DeepHit/Sum_grad/ShapeShapeDeepHit/mul*
T0*
out_type0
О
'DeepHit/gradients/DeepHit/Sum_grad/SizeConst*;
_class1
/-loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape*
dtype0*
value	B :
љ
&DeepHit/gradients/DeepHit/Sum_grad/addAddV2DeepHit/Sum/reduction_indices'DeepHit/gradients/DeepHit/Sum_grad/Size*
T0*;
_class1
/-loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape
…
&DeepHit/gradients/DeepHit/Sum_grad/modFloorMod&DeepHit/gradients/DeepHit/Sum_grad/add'DeepHit/gradients/DeepHit/Sum_grad/Size*
T0*;
_class1
/-loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape
Р
*DeepHit/gradients/DeepHit/Sum_grad/Shape_1Const*;
_class1
/-loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape*
dtype0*
valueB 
Х
.DeepHit/gradients/DeepHit/Sum_grad/range/startConst*;
_class1
/-loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape*
dtype0*
value	B : 
Х
.DeepHit/gradients/DeepHit/Sum_grad/range/deltaConst*;
_class1
/-loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape*
dtype0*
value	B :
Г
(DeepHit/gradients/DeepHit/Sum_grad/rangeRange.DeepHit/gradients/DeepHit/Sum_grad/range/start'DeepHit/gradients/DeepHit/Sum_grad/Size.DeepHit/gradients/DeepHit/Sum_grad/range/delta*

Tidx0*;
_class1
/-loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape
Ф
-DeepHit/gradients/DeepHit/Sum_grad/ones/ConstConst*;
_class1
/-loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape*
dtype0*
value	B :
в
'DeepHit/gradients/DeepHit/Sum_grad/onesFill*DeepHit/gradients/DeepHit/Sum_grad/Shape_1-DeepHit/gradients/DeepHit/Sum_grad/ones/Const*
T0*;
_class1
/-loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape*

index_type0
µ
0DeepHit/gradients/DeepHit/Sum_grad/DynamicStitchDynamicStitch(DeepHit/gradients/DeepHit/Sum_grad/range&DeepHit/gradients/DeepHit/Sum_grad/mod(DeepHit/gradients/DeepHit/Sum_grad/Shape'DeepHit/gradients/DeepHit/Sum_grad/ones*
N*
T0*;
_class1
/-loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape
∞
*DeepHit/gradients/DeepHit/Sum_grad/ReshapeReshape0DeepHit/gradients/DeepHit/Sum_1_grad/BroadcastTo0DeepHit/gradients/DeepHit/Sum_grad/DynamicStitch*
T0*
Tshape0
®
.DeepHit/gradients/DeepHit/Sum_grad/BroadcastToBroadcastTo*DeepHit/gradients/DeepHit/Sum_grad/Reshape(DeepHit/gradients/DeepHit/Sum_grad/Shape*
T0*

Tidx0
[
*DeepHit/gradients/DeepHit/Sum_3_grad/ShapeShapeDeepHit/Sum_2*
T0*
out_type0
њ
0DeepHit/gradients/DeepHit/Sum_3_grad/BroadcastToBroadcastTo=DeepHit/gradients/DeepHit/add_1_grad/tuple/control_dependency*DeepHit/gradients/DeepHit/Sum_3_grad/Shape*
T0*

Tidx0
]
,DeepHit/gradients/DeepHit/truediv_grad/ShapeShapeDeepHit/Neg_1*
T0*
out_type0
W
.DeepHit/gradients/DeepHit/truediv_grad/Shape_1Const*
dtype0*
valueB 
Љ
<DeepHit/gradients/DeepHit/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs,DeepHit/gradients/DeepHit/truediv_grad/Shape.DeepHit/gradients/DeepHit/truediv_grad/Shape_1*
T0
{
.DeepHit/gradients/DeepHit/truediv_grad/RealDivRealDiv&DeepHit/gradients/DeepHit/Exp_grad/mulDeepHit/Const_1*
T0
≈
*DeepHit/gradients/DeepHit/truediv_grad/SumSum.DeepHit/gradients/DeepHit/truediv_grad/RealDiv<DeepHit/gradients/DeepHit/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/truediv_grad/ReshapeReshape*DeepHit/gradients/DeepHit/truediv_grad/Sum,DeepHit/gradients/DeepHit/truediv_grad/Shape*
T0*
Tshape0
I
*DeepHit/gradients/DeepHit/truediv_grad/NegNegDeepHit/Neg_1*
T0
Б
0DeepHit/gradients/DeepHit/truediv_grad/RealDiv_1RealDiv*DeepHit/gradients/DeepHit/truediv_grad/NegDeepHit/Const_1*
T0
З
0DeepHit/gradients/DeepHit/truediv_grad/RealDiv_2RealDiv0DeepHit/gradients/DeepHit/truediv_grad/RealDiv_1DeepHit/Const_1*
T0
Ф
*DeepHit/gradients/DeepHit/truediv_grad/mulMul&DeepHit/gradients/DeepHit/Exp_grad/mul0DeepHit/gradients/DeepHit/truediv_grad/RealDiv_2*
T0
≈
,DeepHit/gradients/DeepHit/truediv_grad/Sum_1Sum*DeepHit/gradients/DeepHit/truediv_grad/mul>DeepHit/gradients/DeepHit/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/truediv_grad/Reshape_1Reshape,DeepHit/gradients/DeepHit/truediv_grad/Sum_1.DeepHit/gradients/DeepHit/truediv_grad/Shape_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/truediv_grad/tuple/group_depsNoOp/^DeepHit/gradients/DeepHit/truediv_grad/Reshape1^DeepHit/gradients/DeepHit/truediv_grad/Reshape_1
Б
?DeepHit/gradients/DeepHit/truediv_grad/tuple/control_dependencyIdentity.DeepHit/gradients/DeepHit/truediv_grad/Reshape8^DeepHit/gradients/DeepHit/truediv_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/truediv_grad/Reshape
З
ADeepHit/gradients/DeepHit/truediv_grad/tuple/control_dependency_1Identity0DeepHit/gradients/DeepHit/truediv_grad/Reshape_18^DeepHit/gradients/DeepHit/truediv_grad/tuple/group_deps*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/truediv_grad/Reshape_1
_
.DeepHit/gradients/DeepHit/truediv_1_grad/ShapeShapeDeepHit/Neg_2*
T0*
out_type0
Y
0DeepHit/gradients/DeepHit/truediv_1_grad/Shape_1Const*
dtype0*
valueB 
¬
>DeepHit/gradients/DeepHit/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs.DeepHit/gradients/DeepHit/truediv_1_grad/Shape0DeepHit/gradients/DeepHit/truediv_1_grad/Shape_1*
T0

0DeepHit/gradients/DeepHit/truediv_1_grad/RealDivRealDiv(DeepHit/gradients/DeepHit/Exp_1_grad/mulDeepHit/Const_1*
T0
Ћ
,DeepHit/gradients/DeepHit/truediv_1_grad/SumSum0DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv>DeepHit/gradients/DeepHit/truediv_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/truediv_1_grad/ReshapeReshape,DeepHit/gradients/DeepHit/truediv_1_grad/Sum.DeepHit/gradients/DeepHit/truediv_1_grad/Shape*
T0*
Tshape0
K
,DeepHit/gradients/DeepHit/truediv_1_grad/NegNegDeepHit/Neg_2*
T0
Е
2DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv_1RealDiv,DeepHit/gradients/DeepHit/truediv_1_grad/NegDeepHit/Const_1*
T0
Л
2DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv_2RealDiv2DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv_1DeepHit/Const_1*
T0
Ъ
,DeepHit/gradients/DeepHit/truediv_1_grad/mulMul(DeepHit/gradients/DeepHit/Exp_1_grad/mul2DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv_2*
T0
Ћ
.DeepHit/gradients/DeepHit/truediv_1_grad/Sum_1Sum,DeepHit/gradients/DeepHit/truediv_1_grad/mul@DeepHit/gradients/DeepHit/truediv_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
ґ
2DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_1Reshape.DeepHit/gradients/DeepHit/truediv_1_grad/Sum_10DeepHit/gradients/DeepHit/truediv_1_grad/Shape_1*
T0*
Tshape0
©
9DeepHit/gradients/DeepHit/truediv_1_grad/tuple/group_depsNoOp1^DeepHit/gradients/DeepHit/truediv_1_grad/Reshape3^DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_1
Й
ADeepHit/gradients/DeepHit/truediv_1_grad/tuple/control_dependencyIdentity0DeepHit/gradients/DeepHit/truediv_1_grad/Reshape:^DeepHit/gradients/DeepHit/truediv_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/truediv_1_grad/Reshape
П
CDeepHit/gradients/DeepHit/truediv_1_grad/tuple/control_dependency_1Identity2DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_1:^DeepHit/gradients/DeepHit/truediv_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_1
_
*DeepHit/gradients/DeepHit/mul_7_grad/ShapeShapeDeepHit/Reshape_7*
T0*
out_type0
]
,DeepHit/gradients/DeepHit/mul_7_grad/Shape_1ShapeDeepHit/mask2*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/mul_7_grad/Shape,DeepHit/gradients/DeepHit/mul_7_grad/Shape_1*
T0
y
(DeepHit/gradients/DeepHit/mul_7_grad/MulMul0DeepHit/gradients/DeepHit/Sum_5_grad/BroadcastToDeepHit/mask2*
T0
ї
(DeepHit/gradients/DeepHit/mul_7_grad/SumSum(DeepHit/gradients/DeepHit/mul_7_grad/Mul:DeepHit/gradients/DeepHit/mul_7_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/mul_7_grad/ReshapeReshape(DeepHit/gradients/DeepHit/mul_7_grad/Sum*DeepHit/gradients/DeepHit/mul_7_grad/Shape*
T0*
Tshape0

*DeepHit/gradients/DeepHit/mul_7_grad/Mul_1MulDeepHit/Reshape_70DeepHit/gradients/DeepHit/Sum_5_grad/BroadcastTo*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_7_grad/Sum_1Sum*DeepHit/gradients/DeepHit/mul_7_grad/Mul_1<DeepHit/gradients/DeepHit/mul_7_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_7_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/mul_7_grad/Sum_1,DeepHit/gradients/DeepHit/mul_7_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/mul_7_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/mul_7_grad/Reshape/^DeepHit/gradients/DeepHit/mul_7_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/mul_7_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/mul_7_grad/Reshape6^DeepHit/gradients/DeepHit/mul_7_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_7_grad/Reshape
€
?DeepHit/gradients/DeepHit/mul_7_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/mul_7_grad/Reshape_16^DeepHit/gradients/DeepHit/mul_7_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_7_grad/Reshape_1
_
*DeepHit/gradients/DeepHit/mul_8_grad/ShapeShapeDeepHit/Reshape_8*
T0*
out_type0
]
,DeepHit/gradients/DeepHit/mul_8_grad/Shape_1ShapeDeepHit/mask2*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/mul_8_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/mul_8_grad/Shape,DeepHit/gradients/DeepHit/mul_8_grad/Shape_1*
T0
y
(DeepHit/gradients/DeepHit/mul_8_grad/MulMul0DeepHit/gradients/DeepHit/Sum_6_grad/BroadcastToDeepHit/mask2*
T0
ї
(DeepHit/gradients/DeepHit/mul_8_grad/SumSum(DeepHit/gradients/DeepHit/mul_8_grad/Mul:DeepHit/gradients/DeepHit/mul_8_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/mul_8_grad/ReshapeReshape(DeepHit/gradients/DeepHit/mul_8_grad/Sum*DeepHit/gradients/DeepHit/mul_8_grad/Shape*
T0*
Tshape0

*DeepHit/gradients/DeepHit/mul_8_grad/Mul_1MulDeepHit/Reshape_80DeepHit/gradients/DeepHit/Sum_6_grad/BroadcastTo*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_8_grad/Sum_1Sum*DeepHit/gradients/DeepHit/mul_8_grad/Mul_1<DeepHit/gradients/DeepHit/mul_8_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_8_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/mul_8_grad/Sum_1,DeepHit/gradients/DeepHit/mul_8_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/mul_8_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/mul_8_grad/Reshape/^DeepHit/gradients/DeepHit/mul_8_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/mul_8_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/mul_8_grad/Reshape6^DeepHit/gradients/DeepHit/mul_8_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_8_grad/Reshape
€
?DeepHit/gradients/DeepHit/mul_8_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/mul_8_grad/Reshape_16^DeepHit/gradients/DeepHit/mul_8_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_8_grad/Reshape_1
Y
(DeepHit/gradients/DeepHit/mul_grad/ShapeShapeDeepHit/mask1*
T0*
out_type0
_
*DeepHit/gradients/DeepHit/mul_grad/Shape_1ShapeDeepHit/Reshape_1*
T0*
out_type0
∞
8DeepHit/gradients/DeepHit/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(DeepHit/gradients/DeepHit/mul_grad/Shape*DeepHit/gradients/DeepHit/mul_grad/Shape_1*
T0
y
&DeepHit/gradients/DeepHit/mul_grad/MulMul.DeepHit/gradients/DeepHit/Sum_grad/BroadcastToDeepHit/Reshape_1*
T0
µ
&DeepHit/gradients/DeepHit/mul_grad/SumSum&DeepHit/gradients/DeepHit/mul_grad/Mul8DeepHit/gradients/DeepHit/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
Ю
*DeepHit/gradients/DeepHit/mul_grad/ReshapeReshape&DeepHit/gradients/DeepHit/mul_grad/Sum(DeepHit/gradients/DeepHit/mul_grad/Shape*
T0*
Tshape0
w
(DeepHit/gradients/DeepHit/mul_grad/Mul_1MulDeepHit/mask1.DeepHit/gradients/DeepHit/Sum_grad/BroadcastTo*
T0
ї
(DeepHit/gradients/DeepHit/mul_grad/Sum_1Sum(DeepHit/gradients/DeepHit/mul_grad/Mul_1:DeepHit/gradients/DeepHit/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/mul_grad/Reshape_1Reshape(DeepHit/gradients/DeepHit/mul_grad/Sum_1*DeepHit/gradients/DeepHit/mul_grad/Shape_1*
T0*
Tshape0
Ч
3DeepHit/gradients/DeepHit/mul_grad/tuple/group_depsNoOp+^DeepHit/gradients/DeepHit/mul_grad/Reshape-^DeepHit/gradients/DeepHit/mul_grad/Reshape_1
с
;DeepHit/gradients/DeepHit/mul_grad/tuple/control_dependencyIdentity*DeepHit/gradients/DeepHit/mul_grad/Reshape4^DeepHit/gradients/DeepHit/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/mul_grad/Reshape
ч
=DeepHit/gradients/DeepHit/mul_grad/tuple/control_dependency_1Identity,DeepHit/gradients/DeepHit/mul_grad/Reshape_14^DeepHit/gradients/DeepHit/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_grad/Reshape_1
[
*DeepHit/gradients/DeepHit/Sum_2_grad/ShapeShapeDeepHit/mul_2*
T0*
out_type0
Т
)DeepHit/gradients/DeepHit/Sum_2_grad/SizeConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape*
dtype0*
value	B :
≈
(DeepHit/gradients/DeepHit/Sum_2_grad/addAddV2DeepHit/Sum_2/reduction_indices)DeepHit/gradients/DeepHit/Sum_2_grad/Size*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape
—
(DeepHit/gradients/DeepHit/Sum_2_grad/modFloorMod(DeepHit/gradients/DeepHit/Sum_2_grad/add)DeepHit/gradients/DeepHit/Sum_2_grad/Size*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape
Ф
,DeepHit/gradients/DeepHit/Sum_2_grad/Shape_1Const*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape*
dtype0*
valueB 
Щ
0DeepHit/gradients/DeepHit/Sum_2_grad/range/startConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape*
dtype0*
value	B : 
Щ
0DeepHit/gradients/DeepHit/Sum_2_grad/range/deltaConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape*
dtype0*
value	B :
Н
*DeepHit/gradients/DeepHit/Sum_2_grad/rangeRange0DeepHit/gradients/DeepHit/Sum_2_grad/range/start)DeepHit/gradients/DeepHit/Sum_2_grad/Size0DeepHit/gradients/DeepHit/Sum_2_grad/range/delta*

Tidx0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape
Ш
/DeepHit/gradients/DeepHit/Sum_2_grad/ones/ConstConst*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape*
dtype0*
value	B :
к
)DeepHit/gradients/DeepHit/Sum_2_grad/onesFill,DeepHit/gradients/DeepHit/Sum_2_grad/Shape_1/DeepHit/gradients/DeepHit/Sum_2_grad/ones/Const*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape*

index_type0
Ѕ
2DeepHit/gradients/DeepHit/Sum_2_grad/DynamicStitchDynamicStitch*DeepHit/gradients/DeepHit/Sum_2_grad/range(DeepHit/gradients/DeepHit/Sum_2_grad/mod*DeepHit/gradients/DeepHit/Sum_2_grad/Shape)DeepHit/gradients/DeepHit/Sum_2_grad/ones*
N*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape
і
,DeepHit/gradients/DeepHit/Sum_2_grad/ReshapeReshape0DeepHit/gradients/DeepHit/Sum_3_grad/BroadcastTo2DeepHit/gradients/DeepHit/Sum_2_grad/DynamicStitch*
T0*
Tshape0
Ѓ
0DeepHit/gradients/DeepHit/Sum_2_grad/BroadcastToBroadcastTo,DeepHit/gradients/DeepHit/Sum_2_grad/Reshape*DeepHit/gradients/DeepHit/Sum_2_grad/Shape*
T0*

Tidx0
y
(DeepHit/gradients/DeepHit/Neg_1_grad/NegNeg?DeepHit/gradients/DeepHit/truediv_grad/tuple/control_dependency*
T0
{
(DeepHit/gradients/DeepHit/Neg_2_grad/NegNegADeepHit/gradients/DeepHit/truediv_1_grad/tuple/control_dependency*
T0
a
.DeepHit/gradients/DeepHit/Reshape_7_grad/ShapeShapeDeepHit/Slice_2*
T0*
out_type0
Ѕ
0DeepHit/gradients/DeepHit/Reshape_7_grad/ReshapeReshape=DeepHit/gradients/DeepHit/mul_7_grad/tuple/control_dependency.DeepHit/gradients/DeepHit/Reshape_7_grad/Shape*
T0*
Tshape0
a
.DeepHit/gradients/DeepHit/Reshape_8_grad/ShapeShapeDeepHit/Slice_3*
T0*
out_type0
Ѕ
0DeepHit/gradients/DeepHit/Reshape_8_grad/ReshapeReshape=DeepHit/gradients/DeepHit/mul_8_grad/tuple/control_dependency.DeepHit/gradients/DeepHit/Reshape_8_grad/Shape*
T0*
Tshape0
[
*DeepHit/gradients/DeepHit/mul_2_grad/ShapeShapeDeepHit/mask1*
T0*
out_type0
a
,DeepHit/gradients/DeepHit/mul_2_grad/Shape_1ShapeDeepHit/Reshape_1*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/mul_2_grad/Shape,DeepHit/gradients/DeepHit/mul_2_grad/Shape_1*
T0
}
(DeepHit/gradients/DeepHit/mul_2_grad/MulMul0DeepHit/gradients/DeepHit/Sum_2_grad/BroadcastToDeepHit/Reshape_1*
T0
ї
(DeepHit/gradients/DeepHit/mul_2_grad/SumSum(DeepHit/gradients/DeepHit/mul_2_grad/Mul:DeepHit/gradients/DeepHit/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/mul_2_grad/ReshapeReshape(DeepHit/gradients/DeepHit/mul_2_grad/Sum*DeepHit/gradients/DeepHit/mul_2_grad/Shape*
T0*
Tshape0
{
*DeepHit/gradients/DeepHit/mul_2_grad/Mul_1MulDeepHit/mask10DeepHit/gradients/DeepHit/Sum_2_grad/BroadcastTo*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_2_grad/Sum_1Sum*DeepHit/gradients/DeepHit/mul_2_grad/Mul_1<DeepHit/gradients/DeepHit/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_2_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/mul_2_grad/Sum_1,DeepHit/gradients/DeepHit/mul_2_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/mul_2_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/mul_2_grad/Reshape/^DeepHit/gradients/DeepHit/mul_2_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/mul_2_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/mul_2_grad/Reshape6^DeepHit/gradients/DeepHit/mul_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_2_grad/Reshape
€
?DeepHit/gradients/DeepHit/mul_2_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/mul_2_grad/Reshape_16^DeepHit/gradients/DeepHit/mul_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_2_grad/Reshape_1
t
<DeepHit/gradients/DeepHit/transpose_2_grad/InvertPermutationInvertPermutationDeepHit/transpose_2/perm*
T0
њ
4DeepHit/gradients/DeepHit/transpose_2_grad/transpose	Transpose(DeepHit/gradients/DeepHit/Neg_1_grad/Neg<DeepHit/gradients/DeepHit/transpose_2_grad/InvertPermutation*
T0*
Tperm0
t
<DeepHit/gradients/DeepHit/transpose_7_grad/InvertPermutationInvertPermutationDeepHit/transpose_7/perm*
T0
њ
4DeepHit/gradients/DeepHit/transpose_7_grad/transpose	Transpose(DeepHit/gradients/DeepHit/Neg_2_grad/Neg<DeepHit/gradients/DeepHit/transpose_7_grad/InvertPermutation*
T0*
Tperm0
U
+DeepHit/gradients/DeepHit/Slice_2_grad/RankConst*
dtype0*
value	B :
_
,DeepHit/gradients/DeepHit/Slice_2_grad/ShapeShapeDeepHit/Slice_2*
T0*
out_type0
X
.DeepHit/gradients/DeepHit/Slice_2_grad/stack/1Const*
dtype0*
value	B :
ѓ
,DeepHit/gradients/DeepHit/Slice_2_grad/stackPack+DeepHit/gradients/DeepHit/Slice_2_grad/Rank.DeepHit/gradients/DeepHit/Slice_2_grad/stack/1*
N*
T0*

axis 
Х
.DeepHit/gradients/DeepHit/Slice_2_grad/ReshapeReshapeDeepHit/Slice_2/begin,DeepHit/gradients/DeepHit/Slice_2_grad/stack*
T0*
Tshape0
c
.DeepHit/gradients/DeepHit/Slice_2_grad/Shape_1ShapeDeepHit/Reshape_1*
T0*
out_type0
Ш
*DeepHit/gradients/DeepHit/Slice_2_grad/subSub.DeepHit/gradients/DeepHit/Slice_2_grad/Shape_1,DeepHit/gradients/DeepHit/Slice_2_grad/Shape*
T0

,DeepHit/gradients/DeepHit/Slice_2_grad/sub_1Sub*DeepHit/gradients/DeepHit/Slice_2_grad/subDeepHit/Slice_2/begin*
T0
Ѓ
0DeepHit/gradients/DeepHit/Slice_2_grad/Reshape_1Reshape,DeepHit/gradients/DeepHit/Slice_2_grad/sub_1,DeepHit/gradients/DeepHit/Slice_2_grad/stack*
T0*
Tshape0
\
2DeepHit/gradients/DeepHit/Slice_2_grad/concat/axisConst*
dtype0*
value	B :
н
-DeepHit/gradients/DeepHit/Slice_2_grad/concatConcatV2.DeepHit/gradients/DeepHit/Slice_2_grad/Reshape0DeepHit/gradients/DeepHit/Slice_2_grad/Reshape_12DeepHit/gradients/DeepHit/Slice_2_grad/concat/axis*
N*
T0*

Tidx0
ђ
*DeepHit/gradients/DeepHit/Slice_2_grad/PadPad0DeepHit/gradients/DeepHit/Reshape_7_grad/Reshape-DeepHit/gradients/DeepHit/Slice_2_grad/concat*
T0*
	Tpaddings0
U
+DeepHit/gradients/DeepHit/Slice_3_grad/RankConst*
dtype0*
value	B :
_
,DeepHit/gradients/DeepHit/Slice_3_grad/ShapeShapeDeepHit/Slice_3*
T0*
out_type0
X
.DeepHit/gradients/DeepHit/Slice_3_grad/stack/1Const*
dtype0*
value	B :
ѓ
,DeepHit/gradients/DeepHit/Slice_3_grad/stackPack+DeepHit/gradients/DeepHit/Slice_3_grad/Rank.DeepHit/gradients/DeepHit/Slice_3_grad/stack/1*
N*
T0*

axis 
Х
.DeepHit/gradients/DeepHit/Slice_3_grad/ReshapeReshapeDeepHit/Slice_3/begin,DeepHit/gradients/DeepHit/Slice_3_grad/stack*
T0*
Tshape0
c
.DeepHit/gradients/DeepHit/Slice_3_grad/Shape_1ShapeDeepHit/Reshape_1*
T0*
out_type0
Ш
*DeepHit/gradients/DeepHit/Slice_3_grad/subSub.DeepHit/gradients/DeepHit/Slice_3_grad/Shape_1,DeepHit/gradients/DeepHit/Slice_3_grad/Shape*
T0

,DeepHit/gradients/DeepHit/Slice_3_grad/sub_1Sub*DeepHit/gradients/DeepHit/Slice_3_grad/subDeepHit/Slice_3/begin*
T0
Ѓ
0DeepHit/gradients/DeepHit/Slice_3_grad/Reshape_1Reshape,DeepHit/gradients/DeepHit/Slice_3_grad/sub_1,DeepHit/gradients/DeepHit/Slice_3_grad/stack*
T0*
Tshape0
\
2DeepHit/gradients/DeepHit/Slice_3_grad/concat/axisConst*
dtype0*
value	B :
н
-DeepHit/gradients/DeepHit/Slice_3_grad/concatConcatV2.DeepHit/gradients/DeepHit/Slice_3_grad/Reshape0DeepHit/gradients/DeepHit/Slice_3_grad/Reshape_12DeepHit/gradients/DeepHit/Slice_3_grad/concat/axis*
N*
T0*

Tidx0
ђ
*DeepHit/gradients/DeepHit/Slice_3_grad/PadPad0DeepHit/gradients/DeepHit/Reshape_8_grad/Reshape-DeepHit/gradients/DeepHit/Slice_3_grad/concat*
T0*
	Tpaddings0
^
*DeepHit/gradients/DeepHit/sub_4_grad/ShapeShapeDeepHit/MatMul_1*
T0*
out_type0
^
,DeepHit/gradients/DeepHit/sub_4_grad/Shape_1ShapeDeepHit/MatMul*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/sub_4_grad/Shape,DeepHit/gradients/DeepHit/sub_4_grad/Shape_1*
T0
«
(DeepHit/gradients/DeepHit/sub_4_grad/SumSum4DeepHit/gradients/DeepHit/transpose_2_grad/transpose:DeepHit/gradients/DeepHit/sub_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/sub_4_grad/ReshapeReshape(DeepHit/gradients/DeepHit/sub_4_grad/Sum*DeepHit/gradients/DeepHit/sub_4_grad/Shape*
T0*
Tshape0
n
(DeepHit/gradients/DeepHit/sub_4_grad/NegNeg4DeepHit/gradients/DeepHit/transpose_2_grad/transpose*
T0
њ
*DeepHit/gradients/DeepHit/sub_4_grad/Sum_1Sum(DeepHit/gradients/DeepHit/sub_4_grad/Neg<DeepHit/gradients/DeepHit/sub_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/sub_4_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/sub_4_grad/Sum_1,DeepHit/gradients/DeepHit/sub_4_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/sub_4_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/sub_4_grad/Reshape/^DeepHit/gradients/DeepHit/sub_4_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/sub_4_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/sub_4_grad/Reshape6^DeepHit/gradients/DeepHit/sub_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/sub_4_grad/Reshape
€
?DeepHit/gradients/DeepHit/sub_4_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/sub_4_grad/Reshape_16^DeepHit/gradients/DeepHit/sub_4_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/sub_4_grad/Reshape_1
^
*DeepHit/gradients/DeepHit/sub_6_grad/ShapeShapeDeepHit/MatMul_6*
T0*
out_type0
`
,DeepHit/gradients/DeepHit/sub_6_grad/Shape_1ShapeDeepHit/MatMul_5*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/sub_6_grad/BroadcastGradientArgsBroadcastGradientArgs*DeepHit/gradients/DeepHit/sub_6_grad/Shape,DeepHit/gradients/DeepHit/sub_6_grad/Shape_1*
T0
«
(DeepHit/gradients/DeepHit/sub_6_grad/SumSum4DeepHit/gradients/DeepHit/transpose_7_grad/transpose:DeepHit/gradients/DeepHit/sub_6_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/sub_6_grad/ReshapeReshape(DeepHit/gradients/DeepHit/sub_6_grad/Sum*DeepHit/gradients/DeepHit/sub_6_grad/Shape*
T0*
Tshape0
n
(DeepHit/gradients/DeepHit/sub_6_grad/NegNeg4DeepHit/gradients/DeepHit/transpose_7_grad/transpose*
T0
њ
*DeepHit/gradients/DeepHit/sub_6_grad/Sum_1Sum(DeepHit/gradients/DeepHit/sub_6_grad/Neg<DeepHit/gradients/DeepHit/sub_6_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/sub_6_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/sub_6_grad/Sum_1,DeepHit/gradients/DeepHit/sub_6_grad/Shape_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/sub_6_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/sub_6_grad/Reshape/^DeepHit/gradients/DeepHit/sub_6_grad/Reshape_1
щ
=DeepHit/gradients/DeepHit/sub_6_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/sub_6_grad/Reshape6^DeepHit/gradients/DeepHit/sub_6_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/sub_6_grad/Reshape
€
?DeepHit/gradients/DeepHit/sub_6_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/sub_6_grad/Reshape_16^DeepHit/gradients/DeepHit/sub_6_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/sub_6_grad/Reshape_1
ї
.DeepHit/gradients/DeepHit/MatMul_1_grad/MatMulMatMul=DeepHit/gradients/DeepHit/sub_4_grad/tuple/control_dependencyDeepHit/transpose_1*
T0*
transpose_a( *
transpose_b(
ї
0DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_1MatMulDeepHit/ones_like=DeepHit/gradients/DeepHit/sub_4_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
§
8DeepHit/gradients/DeepHit/MatMul_1_grad/tuple/group_depsNoOp/^DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul1^DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_1
Г
@DeepHit/gradients/DeepHit/MatMul_1_grad/tuple/control_dependencyIdentity.DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul9^DeepHit/gradients/DeepHit/MatMul_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul
Й
BDeepHit/gradients/DeepHit/MatMul_1_grad/tuple/control_dependency_1Identity0DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_19^DeepHit/gradients/DeepHit/MatMul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_1
ї
.DeepHit/gradients/DeepHit/MatMul_6_grad/MatMulMatMul=DeepHit/gradients/DeepHit/sub_6_grad/tuple/control_dependencyDeepHit/transpose_6*
T0*
transpose_a( *
transpose_b(
љ
0DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_1MatMulDeepHit/ones_like_1=DeepHit/gradients/DeepHit/sub_6_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
§
8DeepHit/gradients/DeepHit/MatMul_6_grad/tuple/group_depsNoOp/^DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul1^DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_1
Г
@DeepHit/gradients/DeepHit/MatMul_6_grad/tuple/control_dependencyIdentity.DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul9^DeepHit/gradients/DeepHit/MatMul_6_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul
Й
BDeepHit/gradients/DeepHit/MatMul_6_grad/tuple/control_dependency_1Identity0DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_19^DeepHit/gradients/DeepHit/MatMul_6_grad/tuple/group_deps*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_1
t
<DeepHit/gradients/DeepHit/transpose_1_grad/InvertPermutationInvertPermutationDeepHit/transpose_1/perm*
T0
ў
4DeepHit/gradients/DeepHit/transpose_1_grad/transpose	TransposeBDeepHit/gradients/DeepHit/MatMul_1_grad/tuple/control_dependency_1<DeepHit/gradients/DeepHit/transpose_1_grad/InvertPermutation*
T0*
Tperm0
t
<DeepHit/gradients/DeepHit/transpose_6_grad/InvertPermutationInvertPermutationDeepHit/transpose_6/perm*
T0
ў
4DeepHit/gradients/DeepHit/transpose_6_grad/transpose	TransposeBDeepHit/gradients/DeepHit/MatMul_6_grad/tuple/control_dependency_1<DeepHit/gradients/DeepHit/transpose_6_grad/InvertPermutation*
T0*
Tperm0
b
.DeepHit/gradients/DeepHit/Reshape_3_grad/ShapeShapeDeepHit/DiagPart*
T0*
out_type0
Є
0DeepHit/gradients/DeepHit/Reshape_3_grad/ReshapeReshape4DeepHit/gradients/DeepHit/transpose_1_grad/transpose.DeepHit/gradients/DeepHit/Reshape_3_grad/Shape*
T0*
Tshape0
d
.DeepHit/gradients/DeepHit/Reshape_5_grad/ShapeShapeDeepHit/DiagPart_1*
T0*
out_type0
Є
0DeepHit/gradients/DeepHit/Reshape_5_grad/ReshapeReshape4DeepHit/gradients/DeepHit/transpose_6_grad/transpose.DeepHit/gradients/DeepHit/Reshape_5_grad/Shape*
T0*
Tshape0
o
,DeepHit/gradients/DeepHit/DiagPart_grad/DiagDiag0DeepHit/gradients/DeepHit/Reshape_3_grad/Reshape*
T0
q
.DeepHit/gradients/DeepHit/DiagPart_1_grad/DiagDiag0DeepHit/gradients/DeepHit/Reshape_5_grad/Reshape*
T0
в
DeepHit/gradients/AddNAddN?DeepHit/gradients/DeepHit/sub_4_grad/tuple/control_dependency_1,DeepHit/gradients/DeepHit/DiagPart_grad/Diag*
N*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/sub_4_grad/Reshape_1
Р
,DeepHit/gradients/DeepHit/MatMul_grad/MatMulMatMulDeepHit/gradients/AddNDeepHit/transpose*
T0*
transpose_a( *
transpose_b(
Т
.DeepHit/gradients/DeepHit/MatMul_grad/MatMul_1MatMulDeepHit/Reshape_2DeepHit/gradients/AddN*
T0*
transpose_a(*
transpose_b( 
Ю
6DeepHit/gradients/DeepHit/MatMul_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/MatMul_grad/MatMul/^DeepHit/gradients/DeepHit/MatMul_grad/MatMul_1
ы
>DeepHit/gradients/DeepHit/MatMul_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/MatMul_grad/MatMul7^DeepHit/gradients/DeepHit/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/MatMul_grad/MatMul
Б
@DeepHit/gradients/DeepHit/MatMul_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/MatMul_grad/MatMul_17^DeepHit/gradients/DeepHit/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/MatMul_grad/MatMul_1
ж
DeepHit/gradients/AddN_1AddN?DeepHit/gradients/DeepHit/sub_6_grad/tuple/control_dependency_1.DeepHit/gradients/DeepHit/DiagPart_1_grad/Diag*
N*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/sub_6_grad/Reshape_1
Ц
.DeepHit/gradients/DeepHit/MatMul_5_grad/MatMulMatMulDeepHit/gradients/AddN_1DeepHit/transpose_5*
T0*
transpose_a( *
transpose_b(
Ц
0DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_1MatMulDeepHit/Reshape_4DeepHit/gradients/AddN_1*
T0*
transpose_a(*
transpose_b( 
§
8DeepHit/gradients/DeepHit/MatMul_5_grad/tuple/group_depsNoOp/^DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul1^DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_1
Г
@DeepHit/gradients/DeepHit/MatMul_5_grad/tuple/control_dependencyIdentity.DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul9^DeepHit/gradients/DeepHit/MatMul_5_grad/tuple/group_deps*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul
Й
BDeepHit/gradients/DeepHit/MatMul_5_grad/tuple/control_dependency_1Identity0DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_19^DeepHit/gradients/DeepHit/MatMul_5_grad/tuple/group_deps*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_1
_
.DeepHit/gradients/DeepHit/Reshape_2_grad/ShapeShapeDeepHit/Slice*
T0*
out_type0
¬
0DeepHit/gradients/DeepHit/Reshape_2_grad/ReshapeReshape>DeepHit/gradients/DeepHit/MatMul_grad/tuple/control_dependency.DeepHit/gradients/DeepHit/Reshape_2_grad/Shape*
T0*
Tshape0
a
.DeepHit/gradients/DeepHit/Reshape_4_grad/ShapeShapeDeepHit/Slice_1*
T0*
out_type0
ƒ
0DeepHit/gradients/DeepHit/Reshape_4_grad/ReshapeReshape@DeepHit/gradients/DeepHit/MatMul_5_grad/tuple/control_dependency.DeepHit/gradients/DeepHit/Reshape_4_grad/Shape*
T0*
Tshape0
S
)DeepHit/gradients/DeepHit/Slice_grad/RankConst*
dtype0*
value	B :
[
*DeepHit/gradients/DeepHit/Slice_grad/ShapeShapeDeepHit/Slice*
T0*
out_type0
V
,DeepHit/gradients/DeepHit/Slice_grad/stack/1Const*
dtype0*
value	B :
©
*DeepHit/gradients/DeepHit/Slice_grad/stackPack)DeepHit/gradients/DeepHit/Slice_grad/Rank,DeepHit/gradients/DeepHit/Slice_grad/stack/1*
N*
T0*

axis 
П
,DeepHit/gradients/DeepHit/Slice_grad/ReshapeReshapeDeepHit/Slice/begin*DeepHit/gradients/DeepHit/Slice_grad/stack*
T0*
Tshape0
a
,DeepHit/gradients/DeepHit/Slice_grad/Shape_1ShapeDeepHit/Reshape_1*
T0*
out_type0
Т
(DeepHit/gradients/DeepHit/Slice_grad/subSub,DeepHit/gradients/DeepHit/Slice_grad/Shape_1*DeepHit/gradients/DeepHit/Slice_grad/Shape*
T0
y
*DeepHit/gradients/DeepHit/Slice_grad/sub_1Sub(DeepHit/gradients/DeepHit/Slice_grad/subDeepHit/Slice/begin*
T0
®
.DeepHit/gradients/DeepHit/Slice_grad/Reshape_1Reshape*DeepHit/gradients/DeepHit/Slice_grad/sub_1*DeepHit/gradients/DeepHit/Slice_grad/stack*
T0*
Tshape0
Z
0DeepHit/gradients/DeepHit/Slice_grad/concat/axisConst*
dtype0*
value	B :
е
+DeepHit/gradients/DeepHit/Slice_grad/concatConcatV2,DeepHit/gradients/DeepHit/Slice_grad/Reshape.DeepHit/gradients/DeepHit/Slice_grad/Reshape_10DeepHit/gradients/DeepHit/Slice_grad/concat/axis*
N*
T0*

Tidx0
®
(DeepHit/gradients/DeepHit/Slice_grad/PadPad0DeepHit/gradients/DeepHit/Reshape_2_grad/Reshape+DeepHit/gradients/DeepHit/Slice_grad/concat*
T0*
	Tpaddings0
U
+DeepHit/gradients/DeepHit/Slice_1_grad/RankConst*
dtype0*
value	B :
_
,DeepHit/gradients/DeepHit/Slice_1_grad/ShapeShapeDeepHit/Slice_1*
T0*
out_type0
X
.DeepHit/gradients/DeepHit/Slice_1_grad/stack/1Const*
dtype0*
value	B :
ѓ
,DeepHit/gradients/DeepHit/Slice_1_grad/stackPack+DeepHit/gradients/DeepHit/Slice_1_grad/Rank.DeepHit/gradients/DeepHit/Slice_1_grad/stack/1*
N*
T0*

axis 
Х
.DeepHit/gradients/DeepHit/Slice_1_grad/ReshapeReshapeDeepHit/Slice_1/begin,DeepHit/gradients/DeepHit/Slice_1_grad/stack*
T0*
Tshape0
c
.DeepHit/gradients/DeepHit/Slice_1_grad/Shape_1ShapeDeepHit/Reshape_1*
T0*
out_type0
Ш
*DeepHit/gradients/DeepHit/Slice_1_grad/subSub.DeepHit/gradients/DeepHit/Slice_1_grad/Shape_1,DeepHit/gradients/DeepHit/Slice_1_grad/Shape*
T0

,DeepHit/gradients/DeepHit/Slice_1_grad/sub_1Sub*DeepHit/gradients/DeepHit/Slice_1_grad/subDeepHit/Slice_1/begin*
T0
Ѓ
0DeepHit/gradients/DeepHit/Slice_1_grad/Reshape_1Reshape,DeepHit/gradients/DeepHit/Slice_1_grad/sub_1,DeepHit/gradients/DeepHit/Slice_1_grad/stack*
T0*
Tshape0
\
2DeepHit/gradients/DeepHit/Slice_1_grad/concat/axisConst*
dtype0*
value	B :
н
-DeepHit/gradients/DeepHit/Slice_1_grad/concatConcatV2.DeepHit/gradients/DeepHit/Slice_1_grad/Reshape0DeepHit/gradients/DeepHit/Slice_1_grad/Reshape_12DeepHit/gradients/DeepHit/Slice_1_grad/concat/axis*
N*
T0*

Tidx0
ђ
*DeepHit/gradients/DeepHit/Slice_1_grad/PadPad0DeepHit/gradients/DeepHit/Reshape_4_grad/Reshape-DeepHit/gradients/DeepHit/Slice_1_grad/concat*
T0*
	Tpaddings0
°
DeepHit/gradients/AddN_2AddN=DeepHit/gradients/DeepHit/mul_grad/tuple/control_dependency_1?DeepHit/gradients/DeepHit/mul_2_grad/tuple/control_dependency_1*DeepHit/gradients/DeepHit/Slice_2_grad/Pad*DeepHit/gradients/DeepHit/Slice_3_grad/Pad(DeepHit/gradients/DeepHit/Slice_grad/Pad*DeepHit/gradients/DeepHit/Slice_1_grad/Pad*
N*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_grad/Reshape_1
h
.DeepHit/gradients/DeepHit/Reshape_1_grad/ShapeShapeDeepHit/Output/Softmax*
T0*
out_type0
Ь
0DeepHit/gradients/DeepHit/Reshape_1_grad/ReshapeReshapeDeepHit/gradients/AddN_2.DeepHit/gradients/DeepHit/Reshape_1_grad/Shape*
T0*
Tshape0
Л
1DeepHit/gradients/DeepHit/Output/Softmax_grad/mulMul0DeepHit/gradients/DeepHit/Reshape_1_grad/ReshapeDeepHit/Output/Softmax*
T0
v
CDeepHit/gradients/DeepHit/Output/Softmax_grad/Sum/reduction_indicesConst*
dtype0*
valueB :
€€€€€€€€€
÷
1DeepHit/gradients/DeepHit/Output/Softmax_grad/SumSum1DeepHit/gradients/DeepHit/Output/Softmax_grad/mulCDeepHit/gradients/DeepHit/Output/Softmax_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims(
¶
1DeepHit/gradients/DeepHit/Output/Softmax_grad/subSub0DeepHit/gradients/DeepHit/Reshape_1_grad/Reshape1DeepHit/gradients/DeepHit/Output/Softmax_grad/Sum*
T0
О
3DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_1Mul1DeepHit/gradients/DeepHit/Output/Softmax_grad/subDeepHit/Output/Softmax*
T0
Э
9DeepHit/gradients/DeepHit/Output/BiasAdd_grad/BiasAddGradBiasAddGrad3DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_1*
T0*
data_formatNHWC
Є
>DeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/group_depsNoOp:^DeepHit/gradients/DeepHit/Output/BiasAdd_grad/BiasAddGrad4^DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_1
Щ
FDeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/control_dependencyIdentity3DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_1?^DeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_1
І
HDeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/control_dependency_1Identity9DeepHit/gradients/DeepHit/Output/BiasAdd_grad/BiasAddGrad?^DeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@DeepHit/gradients/DeepHit/Output/BiasAdd_grad/BiasAddGrad
—
3DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMulMatMulFDeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/control_dependencyDeepHit/Output/weights/read*
T0*
transpose_a( *
transpose_b(
Ќ
5DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_1MatMulDeepHit/dropout_2/MulFDeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
≥
=DeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/group_depsNoOp4^DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul6^DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_1
Ч
EDeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/control_dependencyIdentity3DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul>^DeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul
Э
GDeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/control_dependency_1Identity5DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_1>^DeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_1
o
2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/ShapeShapeDeepHit/dropout_2/RealDiv*
T0*
out_type0
n
4DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape_1ShapeDeepHit/dropout_2/Cast*
T0*
out_type0
ќ
BDeepHit/gradients/DeepHit/dropout_2/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape4DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape_1*
T0
Я
0DeepHit/gradients/DeepHit/dropout_2/Mul_grad/MulMulEDeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/control_dependencyDeepHit/dropout_2/Cast*
T0
”
0DeepHit/gradients/DeepHit/dropout_2/Mul_grad/SumSum0DeepHit/gradients/DeepHit/dropout_2/Mul_grad/MulBDeepHit/gradients/DeepHit/dropout_2/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
Љ
4DeepHit/gradients/DeepHit/dropout_2/Mul_grad/ReshapeReshape0DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Sum2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape*
T0*
Tshape0
§
2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Mul_1MulDeepHit/dropout_2/RealDivEDeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/control_dependency*
T0
ў
2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Sum_1Sum2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Mul_1DDeepHit/gradients/DeepHit/dropout_2/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
¬
6DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_1Reshape2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Sum_14DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape_1*
T0*
Tshape0
µ
=DeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/group_depsNoOp5^DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape7^DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_1
Щ
EDeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/control_dependencyIdentity4DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape>^DeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape
Я
GDeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/control_dependency_1Identity6DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_1>^DeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_1
Т
DeepHit/gradients/AddN_3AddN@DeepHit/gradients/DeepHit/Output/kernel/Regularizer/Abs_grad/mulGDeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/control_dependency_1*
N*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/Output/kernel/Regularizer/Abs_grad/mul
i
6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/ShapeShapeDeepHit/Reshape*
T0*
out_type0
a
8DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape_1Const*
dtype0*
valueB 
Џ
FDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/BroadcastGradientArgsBroadcastGradientArgs6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape8DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape_1*
T0
™
8DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDivRealDivEDeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/control_dependencyDeepHit/dropout_2/Sub*
T0
г
4DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/SumSum8DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDivFDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
»
8DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/ReshapeReshape4DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Sum6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape*
T0*
Tshape0
U
4DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/NegNegDeepHit/Reshape*
T0
Ы
:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDiv_1RealDiv4DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/NegDeepHit/dropout_2/Sub*
T0
°
:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDiv_2RealDiv:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDiv_1DeepHit/dropout_2/Sub*
T0
«
4DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/mulMulEDeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/control_dependency:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDiv_2*
T0
г
6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Sum_1Sum4DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/mulHDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
ќ
:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_1Reshape6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Sum_18DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape_1*
T0*
Tshape0
Ѕ
ADeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/group_depsNoOp9^DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape;^DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_1
©
IDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/control_dependencyIdentity8DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/ReshapeB^DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/group_deps*
T0*K
_classA
?=loc:@DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape
ѓ
KDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/control_dependency_1Identity:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_1B^DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/group_deps*
T0*M
_classC
A?loc:@DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_1
]
,DeepHit/gradients/DeepHit/Reshape_grad/ShapeShapeDeepHit/stack*
T0*
out_type0
…
.DeepHit/gradients/DeepHit/Reshape_grad/ReshapeReshapeIDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/control_dependency,DeepHit/gradients/DeepHit/Reshape_grad/Shape*
T0*
Tshape0
Ж
,DeepHit/gradients/DeepHit/stack_grad/unstackUnpack.DeepHit/gradients/DeepHit/Reshape_grad/Reshape*
T0*

axis*	
num
l
5DeepHit/gradients/DeepHit/stack_grad/tuple/group_depsNoOp-^DeepHit/gradients/DeepHit/stack_grad/unstack
щ
=DeepHit/gradients/DeepHit/stack_grad/tuple/control_dependencyIdentity,DeepHit/gradients/DeepHit/stack_grad/unstack6^DeepHit/gradients/DeepHit/stack_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/stack_grad/unstack
э
?DeepHit/gradients/DeepHit/stack_grad/tuple/control_dependency_1Identity.DeepHit/gradients/DeepHit/stack_grad/unstack:16^DeepHit/gradients/DeepHit/stack_grad/tuple/group_deps*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/stack_grad/unstack
Ѓ
<DeepHit/gradients/DeepHit/fully_connected_3/Elu_grad/EluGradEluGrad=DeepHit/gradients/DeepHit/stack_grad/tuple/control_dependencyDeepHit/fully_connected_3/Elu*
T0
∞
<DeepHit/gradients/DeepHit/fully_connected_4/Elu_grad/EluGradEluGrad?DeepHit/gradients/DeepHit/stack_grad/tuple/control_dependency_1DeepHit/fully_connected_4/Elu*
T0
±
DDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/BiasAddGradBiasAddGrad<DeepHit/gradients/DeepHit/fully_connected_3/Elu_grad/EluGrad*
T0*
data_formatNHWC
„
IDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/group_depsNoOpE^DeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/BiasAddGrad=^DeepHit/gradients/DeepHit/fully_connected_3/Elu_grad/EluGrad
Ѕ
QDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/control_dependencyIdentity<DeepHit/gradients/DeepHit/fully_connected_3/Elu_grad/EluGradJ^DeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/group_deps*
T0*O
_classE
CAloc:@DeepHit/gradients/DeepHit/fully_connected_3/Elu_grad/EluGrad
”
SDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/control_dependency_1IdentityDDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/BiasAddGradJ^DeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@DeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/BiasAddGrad
±
DDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/BiasAddGradBiasAddGrad<DeepHit/gradients/DeepHit/fully_connected_4/Elu_grad/EluGrad*
T0*
data_formatNHWC
„
IDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/group_depsNoOpE^DeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/BiasAddGrad=^DeepHit/gradients/DeepHit/fully_connected_4/Elu_grad/EluGrad
Ѕ
QDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/control_dependencyIdentity<DeepHit/gradients/DeepHit/fully_connected_4/Elu_grad/EluGradJ^DeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/group_deps*
T0*O
_classE
CAloc:@DeepHit/gradients/DeepHit/fully_connected_4/Elu_grad/EluGrad
”
SDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/control_dependency_1IdentityDDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/BiasAddGradJ^DeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@DeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/BiasAddGrad
т
>DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMulMatMulQDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/control_dependency&DeepHit/fully_connected_3/weights/read*
T0*
transpose_a( *
transpose_b(
№
@DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_1MatMulDeepHit/concatQDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
‘
HDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/group_depsNoOp?^DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMulA^DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_1
√
PDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/control_dependencyIdentity>DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMulI^DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul
…
RDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/control_dependency_1Identity@DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_1I^DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_1
т
>DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMulMatMulQDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/control_dependency&DeepHit/fully_connected_4/weights/read*
T0*
transpose_a( *
transpose_b(
№
@DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_1MatMulDeepHit/concatQDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
‘
HDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/group_depsNoOp?^DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMulA^DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_1
√
PDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/control_dependencyIdentity>DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMulI^DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul
…
RDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/control_dependency_1Identity@DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_1I^DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_1
љ
DeepHit/gradients/AddN_4AddNPDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Mul_1RDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/control_dependency_1*
N*
T0*c
_classY
WUloc:@DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Mul_1
©
DeepHit/gradients/AddN_5AddNPDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/control_dependencyPDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/control_dependency*
N*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul
T
*DeepHit/gradients/DeepHit/concat_grad/RankConst*
dtype0*
value	B :

)DeepHit/gradients/DeepHit/concat_grad/modFloorModDeepHit/concat/axis*DeepHit/gradients/DeepHit/concat_grad/Rank*
T0
]
+DeepHit/gradients/DeepHit/concat_grad/ShapeShapeDeepHit/inputs*
T0*
out_type0
З
,DeepHit/gradients/DeepHit/concat_grad/ShapeNShapeNDeepHit/inputsDeepHit/fully_connected_2/Elu*
N*
T0*
out_type0
‘
2DeepHit/gradients/DeepHit/concat_grad/ConcatOffsetConcatOffset)DeepHit/gradients/DeepHit/concat_grad/mod,DeepHit/gradients/DeepHit/concat_grad/ShapeN.DeepHit/gradients/DeepHit/concat_grad/ShapeN:1*
N
∆
+DeepHit/gradients/DeepHit/concat_grad/SliceSliceDeepHit/gradients/AddN_52DeepHit/gradients/DeepHit/concat_grad/ConcatOffset,DeepHit/gradients/DeepHit/concat_grad/ShapeN*
Index0*
T0
ћ
-DeepHit/gradients/DeepHit/concat_grad/Slice_1SliceDeepHit/gradients/AddN_54DeepHit/gradients/DeepHit/concat_grad/ConcatOffset:1.DeepHit/gradients/DeepHit/concat_grad/ShapeN:1*
Index0*
T0
Ь
6DeepHit/gradients/DeepHit/concat_grad/tuple/group_depsNoOp,^DeepHit/gradients/DeepHit/concat_grad/Slice.^DeepHit/gradients/DeepHit/concat_grad/Slice_1
щ
>DeepHit/gradients/DeepHit/concat_grad/tuple/control_dependencyIdentity+DeepHit/gradients/DeepHit/concat_grad/Slice7^DeepHit/gradients/DeepHit/concat_grad/tuple/group_deps*
T0*>
_class4
20loc:@DeepHit/gradients/DeepHit/concat_grad/Slice
€
@DeepHit/gradients/DeepHit/concat_grad/tuple/control_dependency_1Identity-DeepHit/gradients/DeepHit/concat_grad/Slice_17^DeepHit/gradients/DeepHit/concat_grad/tuple/group_deps*
T0*@
_class6
42loc:@DeepHit/gradients/DeepHit/concat_grad/Slice_1
љ
DeepHit/gradients/AddN_6AddNPDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Mul_1RDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/control_dependency_1*
N*
T0*c
_classY
WUloc:@DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Mul_1
±
<DeepHit/gradients/DeepHit/fully_connected_2/Elu_grad/EluGradEluGrad@DeepHit/gradients/DeepHit/concat_grad/tuple/control_dependency_1DeepHit/fully_connected_2/Elu*
T0
±
DDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/BiasAddGradBiasAddGrad<DeepHit/gradients/DeepHit/fully_connected_2/Elu_grad/EluGrad*
T0*
data_formatNHWC
„
IDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/group_depsNoOpE^DeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/BiasAddGrad=^DeepHit/gradients/DeepHit/fully_connected_2/Elu_grad/EluGrad
Ѕ
QDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/control_dependencyIdentity<DeepHit/gradients/DeepHit/fully_connected_2/Elu_grad/EluGradJ^DeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/group_deps*
T0*O
_classE
CAloc:@DeepHit/gradients/DeepHit/fully_connected_2/Elu_grad/EluGrad
”
SDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1IdentityDDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/BiasAddGradJ^DeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@DeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/BiasAddGrad
т
>DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMulMatMulQDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/control_dependency&DeepHit/fully_connected_2/weights/read*
T0*
transpose_a( *
transpose_b(
г
@DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_1MatMulDeepHit/dropout_1/MulQDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
‘
HDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/group_depsNoOp?^DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMulA^DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_1
√
PDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/control_dependencyIdentity>DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMulI^DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul
…
RDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/control_dependency_1Identity@DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_1I^DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_1
o
2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/ShapeShapeDeepHit/dropout_1/RealDiv*
T0*
out_type0
n
4DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape_1ShapeDeepHit/dropout_1/Cast*
T0*
out_type0
ќ
BDeepHit/gradients/DeepHit/dropout_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape4DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape_1*
T0
™
0DeepHit/gradients/DeepHit/dropout_1/Mul_grad/MulMulPDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/control_dependencyDeepHit/dropout_1/Cast*
T0
”
0DeepHit/gradients/DeepHit/dropout_1/Mul_grad/SumSum0DeepHit/gradients/DeepHit/dropout_1/Mul_grad/MulBDeepHit/gradients/DeepHit/dropout_1/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
Љ
4DeepHit/gradients/DeepHit/dropout_1/Mul_grad/ReshapeReshape0DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Sum2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape*
T0*
Tshape0
ѓ
2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Mul_1MulDeepHit/dropout_1/RealDivPDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/control_dependency*
T0
ў
2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Sum_1Sum2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Mul_1DDeepHit/gradients/DeepHit/dropout_1/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
¬
6DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_1Reshape2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Sum_14DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape_1*
T0*
Tshape0
µ
=DeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/group_depsNoOp5^DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape7^DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_1
Щ
EDeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/control_dependencyIdentity4DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape>^DeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape
Я
GDeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/control_dependency_1Identity6DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_1>^DeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_1
љ
DeepHit/gradients/AddN_7AddNPDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Mul_1RDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/control_dependency_1*
N*
T0*c
_classY
WUloc:@DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Mul_1
w
6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/ShapeShapeDeepHit/fully_connected_1/Elu*
T0*
out_type0
a
8DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape_1Const*
dtype0*
valueB 
Џ
FDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/BroadcastGradientArgsBroadcastGradientArgs6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape8DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape_1*
T0
™
8DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDivRealDivEDeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/control_dependencyDeepHit/dropout_1/Sub*
T0
г
4DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/SumSum8DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDivFDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
»
8DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/ReshapeReshape4DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Sum6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape*
T0*
Tshape0
c
4DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/NegNegDeepHit/fully_connected_1/Elu*
T0
Ы
:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDiv_1RealDiv4DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/NegDeepHit/dropout_1/Sub*
T0
°
:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDiv_2RealDiv:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDiv_1DeepHit/dropout_1/Sub*
T0
«
4DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/mulMulEDeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/control_dependency:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDiv_2*
T0
г
6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Sum_1Sum4DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/mulHDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
ќ
:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_1Reshape6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Sum_18DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape_1*
T0*
Tshape0
Ѕ
ADeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/group_depsNoOp9^DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape;^DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_1
©
IDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/control_dependencyIdentity8DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/ReshapeB^DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/group_deps*
T0*K
_classA
?=loc:@DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape
ѓ
KDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/control_dependency_1Identity:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_1B^DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/group_deps*
T0*M
_classC
A?loc:@DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_1
Ї
<DeepHit/gradients/DeepHit/fully_connected_1/Elu_grad/EluGradEluGradIDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/control_dependencyDeepHit/fully_connected_1/Elu*
T0
±
DDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGrad<DeepHit/gradients/DeepHit/fully_connected_1/Elu_grad/EluGrad*
T0*
data_formatNHWC
„
IDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOpE^DeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/BiasAddGrad=^DeepHit/gradients/DeepHit/fully_connected_1/Elu_grad/EluGrad
Ѕ
QDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentity<DeepHit/gradients/DeepHit/fully_connected_1/Elu_grad/EluGradJ^DeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/group_deps*
T0*O
_classE
CAloc:@DeepHit/gradients/DeepHit/fully_connected_1/Elu_grad/EluGrad
”
SDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1IdentityDDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/BiasAddGradJ^DeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@DeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/BiasAddGrad
т
>DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMulMatMulQDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/control_dependency&DeepHit/fully_connected_1/weights/read*
T0*
transpose_a( *
transpose_b(
б
@DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_1MatMulDeepHit/dropout/MulQDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
‘
HDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/group_depsNoOp?^DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMulA^DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_1
√
PDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity>DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMulI^DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul
…
RDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity@DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_1I^DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_1
k
0DeepHit/gradients/DeepHit/dropout/Mul_grad/ShapeShapeDeepHit/dropout/RealDiv*
T0*
out_type0
j
2DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape_1ShapeDeepHit/dropout/Cast*
T0*
out_type0
»
@DeepHit/gradients/DeepHit/dropout/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs0DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape2DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape_1*
T0
¶
.DeepHit/gradients/DeepHit/dropout/Mul_grad/MulMulPDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/control_dependencyDeepHit/dropout/Cast*
T0
Ќ
.DeepHit/gradients/DeepHit/dropout/Mul_grad/SumSum.DeepHit/gradients/DeepHit/dropout/Mul_grad/Mul@DeepHit/gradients/DeepHit/dropout/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
ґ
2DeepHit/gradients/DeepHit/dropout/Mul_grad/ReshapeReshape.DeepHit/gradients/DeepHit/dropout/Mul_grad/Sum0DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape*
T0*
Tshape0
Ђ
0DeepHit/gradients/DeepHit/dropout/Mul_grad/Mul_1MulDeepHit/dropout/RealDivPDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/control_dependency*
T0
”
0DeepHit/gradients/DeepHit/dropout/Mul_grad/Sum_1Sum0DeepHit/gradients/DeepHit/dropout/Mul_grad/Mul_1BDeepHit/gradients/DeepHit/dropout/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
Љ
4DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_1Reshape0DeepHit/gradients/DeepHit/dropout/Mul_grad/Sum_12DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape_1*
T0*
Tshape0
ѓ
;DeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/group_depsNoOp3^DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape5^DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_1
С
CDeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/control_dependencyIdentity2DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape<^DeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape
Ч
EDeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/control_dependency_1Identity4DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_1<^DeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_1
љ
DeepHit/gradients/AddN_8AddNPDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Mul_1RDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
N*
T0*c
_classY
WUloc:@DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Mul_1
s
4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/ShapeShapeDeepHit/fully_connected/Elu*
T0*
out_type0
_
6DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape_1Const*
dtype0*
valueB 
‘
DDeepHit/gradients/DeepHit/dropout/RealDiv_grad/BroadcastGradientArgsBroadcastGradientArgs4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape6DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape_1*
T0
§
6DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDivRealDivCDeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/control_dependencyDeepHit/dropout/Sub*
T0
Ё
2DeepHit/gradients/DeepHit/dropout/RealDiv_grad/SumSum6DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDivDDeepHit/gradients/DeepHit/dropout/RealDiv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
¬
6DeepHit/gradients/DeepHit/dropout/RealDiv_grad/ReshapeReshape2DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Sum4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape*
T0*
Tshape0
_
2DeepHit/gradients/DeepHit/dropout/RealDiv_grad/NegNegDeepHit/fully_connected/Elu*
T0
Х
8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDiv_1RealDiv2DeepHit/gradients/DeepHit/dropout/RealDiv_grad/NegDeepHit/dropout/Sub*
T0
Ы
8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDiv_2RealDiv8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDiv_1DeepHit/dropout/Sub*
T0
Ѕ
2DeepHit/gradients/DeepHit/dropout/RealDiv_grad/mulMulCDeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/control_dependency8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDiv_2*
T0
Ё
4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Sum_1Sum2DeepHit/gradients/DeepHit/dropout/RealDiv_grad/mulFDeepHit/gradients/DeepHit/dropout/RealDiv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
»
8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_1Reshape4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Sum_16DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape_1*
T0*
Tshape0
ї
?DeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/group_depsNoOp7^DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape9^DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_1
°
GDeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/control_dependencyIdentity6DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape@^DeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/group_deps*
T0*I
_class?
=;loc:@DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape
І
IDeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/control_dependency_1Identity8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_1@^DeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/group_deps*
T0*K
_classA
?=loc:@DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_1
і
:DeepHit/gradients/DeepHit/fully_connected/Elu_grad/EluGradEluGradGDeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/control_dependencyDeepHit/fully_connected/Elu*
T0
≠
BDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad:DeepHit/gradients/DeepHit/fully_connected/Elu_grad/EluGrad*
T0*
data_formatNHWC
—
GDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/group_depsNoOpC^DeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/BiasAddGrad;^DeepHit/gradients/DeepHit/fully_connected/Elu_grad/EluGrad
є
ODeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity:DeepHit/gradients/DeepHit/fully_connected/Elu_grad/EluGradH^DeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@DeepHit/gradients/DeepHit/fully_connected/Elu_grad/EluGrad
Ћ
QDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityBDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/BiasAddGradH^DeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@DeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/BiasAddGrad
м
<DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMulMatMulODeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/control_dependency$DeepHit/fully_connected/weights/read*
T0*
transpose_a( *
transpose_b(
Ў
>DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_1MatMulDeepHit/inputsODeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
ќ
FDeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/group_depsNoOp=^DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul?^DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_1
ї
NDeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/control_dependencyIdentity<DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMulG^DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul
Ѕ
PDeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/control_dependency_1Identity>DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_1G^DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_1
Ј
DeepHit/gradients/AddN_9AddNNDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Mul_1PDeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/control_dependency_1*
N*
T0*a
_classW
USloc:@DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Mul_1
x
!DeepHit/beta1_power/initial_valueConst*(
_class
loc:@DeepHit/Output/biases*
dtype0*
valueB
 *fff?
Й
DeepHit/beta1_power
VariableV2*(
_class
loc:@DeepHit/Output/biases*
	container *
dtype0*
shape: *
shared_name 
Є
DeepHit/beta1_power/AssignAssignDeepHit/beta1_power!DeepHit/beta1_power/initial_value*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking(*
validate_shape(
l
DeepHit/beta1_power/readIdentityDeepHit/beta1_power*
T0*(
_class
loc:@DeepHit/Output/biases
x
!DeepHit/beta2_power/initial_valueConst*(
_class
loc:@DeepHit/Output/biases*
dtype0*
valueB
 *wЊ?
Й
DeepHit/beta2_power
VariableV2*(
_class
loc:@DeepHit/Output/biases*
	container *
dtype0*
shape: *
shared_name 
Є
DeepHit/beta2_power/AssignAssignDeepHit/beta2_power!DeepHit/beta2_power/initial_value*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking(*
validate_shape(
l
DeepHit/beta2_power/readIdentityDeepHit/beta2_power*
T0*(
_class
loc:@DeepHit/Output/biases
І
>DeepHit/DeepHit/fully_connected/weights/Adam/Initializer/zerosConst*2
_class(
&$loc:@DeepHit/fully_connected/weights*
dtype0*
valueB`
*    
і
,DeepHit/DeepHit/fully_connected/weights/Adam
VariableV2*2
_class(
&$loc:@DeepHit/fully_connected/weights*
	container *
dtype0*
shape
:`
*
shared_name 
С
3DeepHit/DeepHit/fully_connected/weights/Adam/AssignAssign,DeepHit/DeepHit/fully_connected/weights/Adam>DeepHit/DeepHit/fully_connected/weights/Adam/Initializer/zeros*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights*
use_locking(*
validate_shape(
®
1DeepHit/DeepHit/fully_connected/weights/Adam/readIdentity,DeepHit/DeepHit/fully_connected/weights/Adam*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights
©
@DeepHit/DeepHit/fully_connected/weights/Adam_1/Initializer/zerosConst*2
_class(
&$loc:@DeepHit/fully_connected/weights*
dtype0*
valueB`
*    
ґ
.DeepHit/DeepHit/fully_connected/weights/Adam_1
VariableV2*2
_class(
&$loc:@DeepHit/fully_connected/weights*
	container *
dtype0*
shape
:`
*
shared_name 
Ч
5DeepHit/DeepHit/fully_connected/weights/Adam_1/AssignAssign.DeepHit/DeepHit/fully_connected/weights/Adam_1@DeepHit/DeepHit/fully_connected/weights/Adam_1/Initializer/zeros*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights*
use_locking(*
validate_shape(
ђ
3DeepHit/DeepHit/fully_connected/weights/Adam_1/readIdentity.DeepHit/DeepHit/fully_connected/weights/Adam_1*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights
°
=DeepHit/DeepHit/fully_connected/biases/Adam/Initializer/zerosConst*1
_class'
%#loc:@DeepHit/fully_connected/biases*
dtype0*
valueB
*    
Ѓ
+DeepHit/DeepHit/fully_connected/biases/Adam
VariableV2*1
_class'
%#loc:@DeepHit/fully_connected/biases*
	container *
dtype0*
shape:
*
shared_name 
Н
2DeepHit/DeepHit/fully_connected/biases/Adam/AssignAssign+DeepHit/DeepHit/fully_connected/biases/Adam=DeepHit/DeepHit/fully_connected/biases/Adam/Initializer/zeros*
T0*1
_class'
%#loc:@DeepHit/fully_connected/biases*
use_locking(*
validate_shape(
•
0DeepHit/DeepHit/fully_connected/biases/Adam/readIdentity+DeepHit/DeepHit/fully_connected/biases/Adam*
T0*1
_class'
%#loc:@DeepHit/fully_connected/biases
£
?DeepHit/DeepHit/fully_connected/biases/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@DeepHit/fully_connected/biases*
dtype0*
valueB
*    
∞
-DeepHit/DeepHit/fully_connected/biases/Adam_1
VariableV2*1
_class'
%#loc:@DeepHit/fully_connected/biases*
	container *
dtype0*
shape:
*
shared_name 
У
4DeepHit/DeepHit/fully_connected/biases/Adam_1/AssignAssign-DeepHit/DeepHit/fully_connected/biases/Adam_1?DeepHit/DeepHit/fully_connected/biases/Adam_1/Initializer/zeros*
T0*1
_class'
%#loc:@DeepHit/fully_connected/biases*
use_locking(*
validate_shape(
©
2DeepHit/DeepHit/fully_connected/biases/Adam_1/readIdentity-DeepHit/DeepHit/fully_connected/biases/Adam_1*
T0*1
_class'
%#loc:@DeepHit/fully_connected/biases
Ђ
@DeepHit/DeepHit/fully_connected_1/weights/Adam/Initializer/zerosConst*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
dtype0*
valueB

*    
Є
.DeepHit/DeepHit/fully_connected_1/weights/Adam
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
	container *
dtype0*
shape
:

*
shared_name 
Щ
5DeepHit/DeepHit/fully_connected_1/weights/Adam/AssignAssign.DeepHit/DeepHit/fully_connected_1/weights/Adam@DeepHit/DeepHit/fully_connected_1/weights/Adam/Initializer/zeros*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
use_locking(*
validate_shape(
Ѓ
3DeepHit/DeepHit/fully_connected_1/weights/Adam/readIdentity.DeepHit/DeepHit/fully_connected_1/weights/Adam*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights
≠
BDeepHit/DeepHit/fully_connected_1/weights/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
dtype0*
valueB

*    
Ї
0DeepHit/DeepHit/fully_connected_1/weights/Adam_1
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
	container *
dtype0*
shape
:

*
shared_name 
Я
7DeepHit/DeepHit/fully_connected_1/weights/Adam_1/AssignAssign0DeepHit/DeepHit/fully_connected_1/weights/Adam_1BDeepHit/DeepHit/fully_connected_1/weights/Adam_1/Initializer/zeros*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
use_locking(*
validate_shape(
≤
5DeepHit/DeepHit/fully_connected_1/weights/Adam_1/readIdentity0DeepHit/DeepHit/fully_connected_1/weights/Adam_1*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights
•
?DeepHit/DeepHit/fully_connected_1/biases/Adam/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
dtype0*
valueB
*    
≤
-DeepHit/DeepHit/fully_connected_1/biases/Adam
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
	container *
dtype0*
shape:
*
shared_name 
Х
4DeepHit/DeepHit/fully_connected_1/biases/Adam/AssignAssign-DeepHit/DeepHit/fully_connected_1/biases/Adam?DeepHit/DeepHit/fully_connected_1/biases/Adam/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
use_locking(*
validate_shape(
Ђ
2DeepHit/DeepHit/fully_connected_1/biases/Adam/readIdentity-DeepHit/DeepHit/fully_connected_1/biases/Adam*
T0*3
_class)
'%loc:@DeepHit/fully_connected_1/biases
І
ADeepHit/DeepHit/fully_connected_1/biases/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
dtype0*
valueB
*    
і
/DeepHit/DeepHit/fully_connected_1/biases/Adam_1
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
	container *
dtype0*
shape:
*
shared_name 
Ы
6DeepHit/DeepHit/fully_connected_1/biases/Adam_1/AssignAssign/DeepHit/DeepHit/fully_connected_1/biases/Adam_1ADeepHit/DeepHit/fully_connected_1/biases/Adam_1/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
use_locking(*
validate_shape(
ѓ
4DeepHit/DeepHit/fully_connected_1/biases/Adam_1/readIdentity/DeepHit/DeepHit/fully_connected_1/biases/Adam_1*
T0*3
_class)
'%loc:@DeepHit/fully_connected_1/biases
Ђ
@DeepHit/DeepHit/fully_connected_2/weights/Adam/Initializer/zerosConst*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
dtype0*
valueB

*    
Є
.DeepHit/DeepHit/fully_connected_2/weights/Adam
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
	container *
dtype0*
shape
:

*
shared_name 
Щ
5DeepHit/DeepHit/fully_connected_2/weights/Adam/AssignAssign.DeepHit/DeepHit/fully_connected_2/weights/Adam@DeepHit/DeepHit/fully_connected_2/weights/Adam/Initializer/zeros*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
use_locking(*
validate_shape(
Ѓ
3DeepHit/DeepHit/fully_connected_2/weights/Adam/readIdentity.DeepHit/DeepHit/fully_connected_2/weights/Adam*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights
≠
BDeepHit/DeepHit/fully_connected_2/weights/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
dtype0*
valueB

*    
Ї
0DeepHit/DeepHit/fully_connected_2/weights/Adam_1
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
	container *
dtype0*
shape
:

*
shared_name 
Я
7DeepHit/DeepHit/fully_connected_2/weights/Adam_1/AssignAssign0DeepHit/DeepHit/fully_connected_2/weights/Adam_1BDeepHit/DeepHit/fully_connected_2/weights/Adam_1/Initializer/zeros*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
use_locking(*
validate_shape(
≤
5DeepHit/DeepHit/fully_connected_2/weights/Adam_1/readIdentity0DeepHit/DeepHit/fully_connected_2/weights/Adam_1*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights
•
?DeepHit/DeepHit/fully_connected_2/biases/Adam/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
dtype0*
valueB
*    
≤
-DeepHit/DeepHit/fully_connected_2/biases/Adam
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
	container *
dtype0*
shape:
*
shared_name 
Х
4DeepHit/DeepHit/fully_connected_2/biases/Adam/AssignAssign-DeepHit/DeepHit/fully_connected_2/biases/Adam?DeepHit/DeepHit/fully_connected_2/biases/Adam/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
use_locking(*
validate_shape(
Ђ
2DeepHit/DeepHit/fully_connected_2/biases/Adam/readIdentity-DeepHit/DeepHit/fully_connected_2/biases/Adam*
T0*3
_class)
'%loc:@DeepHit/fully_connected_2/biases
І
ADeepHit/DeepHit/fully_connected_2/biases/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
dtype0*
valueB
*    
і
/DeepHit/DeepHit/fully_connected_2/biases/Adam_1
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
	container *
dtype0*
shape:
*
shared_name 
Ы
6DeepHit/DeepHit/fully_connected_2/biases/Adam_1/AssignAssign/DeepHit/DeepHit/fully_connected_2/biases/Adam_1ADeepHit/DeepHit/fully_connected_2/biases/Adam_1/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
use_locking(*
validate_shape(
ѓ
4DeepHit/DeepHit/fully_connected_2/biases/Adam_1/readIdentity/DeepHit/DeepHit/fully_connected_2/biases/Adam_1*
T0*3
_class)
'%loc:@DeepHit/fully_connected_2/biases
ї
PDeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros/shape_as_tensorConst*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
dtype0*
valueB"j      
©
FDeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros/ConstConst*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
dtype0*
valueB
 *    
≥
@DeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zerosFillPDeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros/shape_as_tensorFDeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros/Const*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*

index_type0
Є
.DeepHit/DeepHit/fully_connected_3/weights/Adam
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
	container *
dtype0*
shape
:j*
shared_name 
Щ
5DeepHit/DeepHit/fully_connected_3/weights/Adam/AssignAssign.DeepHit/DeepHit/fully_connected_3/weights/Adam@DeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
use_locking(*
validate_shape(
Ѓ
3DeepHit/DeepHit/fully_connected_3/weights/Adam/readIdentity.DeepHit/DeepHit/fully_connected_3/weights/Adam*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights
љ
RDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
dtype0*
valueB"j      
Ђ
HDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros/ConstConst*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
dtype0*
valueB
 *    
є
BDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zerosFillRDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros/shape_as_tensorHDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros/Const*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*

index_type0
Ї
0DeepHit/DeepHit/fully_connected_3/weights/Adam_1
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
	container *
dtype0*
shape
:j*
shared_name 
Я
7DeepHit/DeepHit/fully_connected_3/weights/Adam_1/AssignAssign0DeepHit/DeepHit/fully_connected_3/weights/Adam_1BDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
use_locking(*
validate_shape(
≤
5DeepHit/DeepHit/fully_connected_3/weights/Adam_1/readIdentity0DeepHit/DeepHit/fully_connected_3/weights/Adam_1*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights
•
?DeepHit/DeepHit/fully_connected_3/biases/Adam/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
dtype0*
valueB*    
≤
-DeepHit/DeepHit/fully_connected_3/biases/Adam
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
	container *
dtype0*
shape:*
shared_name 
Х
4DeepHit/DeepHit/fully_connected_3/biases/Adam/AssignAssign-DeepHit/DeepHit/fully_connected_3/biases/Adam?DeepHit/DeepHit/fully_connected_3/biases/Adam/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
use_locking(*
validate_shape(
Ђ
2DeepHit/DeepHit/fully_connected_3/biases/Adam/readIdentity-DeepHit/DeepHit/fully_connected_3/biases/Adam*
T0*3
_class)
'%loc:@DeepHit/fully_connected_3/biases
І
ADeepHit/DeepHit/fully_connected_3/biases/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
dtype0*
valueB*    
і
/DeepHit/DeepHit/fully_connected_3/biases/Adam_1
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
	container *
dtype0*
shape:*
shared_name 
Ы
6DeepHit/DeepHit/fully_connected_3/biases/Adam_1/AssignAssign/DeepHit/DeepHit/fully_connected_3/biases/Adam_1ADeepHit/DeepHit/fully_connected_3/biases/Adam_1/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
use_locking(*
validate_shape(
ѓ
4DeepHit/DeepHit/fully_connected_3/biases/Adam_1/readIdentity/DeepHit/DeepHit/fully_connected_3/biases/Adam_1*
T0*3
_class)
'%loc:@DeepHit/fully_connected_3/biases
ї
PDeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros/shape_as_tensorConst*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
dtype0*
valueB"j      
©
FDeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros/ConstConst*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
dtype0*
valueB
 *    
≥
@DeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zerosFillPDeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros/shape_as_tensorFDeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros/Const*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*

index_type0
Є
.DeepHit/DeepHit/fully_connected_4/weights/Adam
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
	container *
dtype0*
shape
:j*
shared_name 
Щ
5DeepHit/DeepHit/fully_connected_4/weights/Adam/AssignAssign.DeepHit/DeepHit/fully_connected_4/weights/Adam@DeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
use_locking(*
validate_shape(
Ѓ
3DeepHit/DeepHit/fully_connected_4/weights/Adam/readIdentity.DeepHit/DeepHit/fully_connected_4/weights/Adam*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights
љ
RDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
dtype0*
valueB"j      
Ђ
HDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros/ConstConst*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
dtype0*
valueB
 *    
є
BDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zerosFillRDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros/shape_as_tensorHDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros/Const*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*

index_type0
Ї
0DeepHit/DeepHit/fully_connected_4/weights/Adam_1
VariableV2*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
	container *
dtype0*
shape
:j*
shared_name 
Я
7DeepHit/DeepHit/fully_connected_4/weights/Adam_1/AssignAssign0DeepHit/DeepHit/fully_connected_4/weights/Adam_1BDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
use_locking(*
validate_shape(
≤
5DeepHit/DeepHit/fully_connected_4/weights/Adam_1/readIdentity0DeepHit/DeepHit/fully_connected_4/weights/Adam_1*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights
•
?DeepHit/DeepHit/fully_connected_4/biases/Adam/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
dtype0*
valueB*    
≤
-DeepHit/DeepHit/fully_connected_4/biases/Adam
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
	container *
dtype0*
shape:*
shared_name 
Х
4DeepHit/DeepHit/fully_connected_4/biases/Adam/AssignAssign-DeepHit/DeepHit/fully_connected_4/biases/Adam?DeepHit/DeepHit/fully_connected_4/biases/Adam/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
use_locking(*
validate_shape(
Ђ
2DeepHit/DeepHit/fully_connected_4/biases/Adam/readIdentity-DeepHit/DeepHit/fully_connected_4/biases/Adam*
T0*3
_class)
'%loc:@DeepHit/fully_connected_4/biases
І
ADeepHit/DeepHit/fully_connected_4/biases/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
dtype0*
valueB*    
і
/DeepHit/DeepHit/fully_connected_4/biases/Adam_1
VariableV2*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
	container *
dtype0*
shape:*
shared_name 
Ы
6DeepHit/DeepHit/fully_connected_4/biases/Adam_1/AssignAssign/DeepHit/DeepHit/fully_connected_4/biases/Adam_1ADeepHit/DeepHit/fully_connected_4/biases/Adam_1/Initializer/zeros*
T0*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
use_locking(*
validate_shape(
ѓ
4DeepHit/DeepHit/fully_connected_4/biases/Adam_1/readIdentity/DeepHit/DeepHit/fully_connected_4/biases/Adam_1*
T0*3
_class)
'%loc:@DeepHit/fully_connected_4/biases
•
EDeepHit/DeepHit/Output/weights/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@DeepHit/Output/weights*
dtype0*
valueB"(     
У
;DeepHit/DeepHit/Output/weights/Adam/Initializer/zeros/ConstConst*)
_class
loc:@DeepHit/Output/weights*
dtype0*
valueB
 *    
З
5DeepHit/DeepHit/Output/weights/Adam/Initializer/zerosFillEDeepHit/DeepHit/Output/weights/Adam/Initializer/zeros/shape_as_tensor;DeepHit/DeepHit/Output/weights/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@DeepHit/Output/weights*

index_type0
£
#DeepHit/DeepHit/Output/weights/Adam
VariableV2*)
_class
loc:@DeepHit/Output/weights*
	container *
dtype0*
shape:	(Ю*
shared_name 
н
*DeepHit/DeepHit/Output/weights/Adam/AssignAssign#DeepHit/DeepHit/Output/weights/Adam5DeepHit/DeepHit/Output/weights/Adam/Initializer/zeros*
T0*)
_class
loc:@DeepHit/Output/weights*
use_locking(*
validate_shape(
Н
(DeepHit/DeepHit/Output/weights/Adam/readIdentity#DeepHit/DeepHit/Output/weights/Adam*
T0*)
_class
loc:@DeepHit/Output/weights
І
GDeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@DeepHit/Output/weights*
dtype0*
valueB"(     
Х
=DeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@DeepHit/Output/weights*
dtype0*
valueB
 *    
Н
7DeepHit/DeepHit/Output/weights/Adam_1/Initializer/zerosFillGDeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros/shape_as_tensor=DeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@DeepHit/Output/weights*

index_type0
•
%DeepHit/DeepHit/Output/weights/Adam_1
VariableV2*)
_class
loc:@DeepHit/Output/weights*
	container *
dtype0*
shape:	(Ю*
shared_name 
у
,DeepHit/DeepHit/Output/weights/Adam_1/AssignAssign%DeepHit/DeepHit/Output/weights/Adam_17DeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros*
T0*)
_class
loc:@DeepHit/Output/weights*
use_locking(*
validate_shape(
С
*DeepHit/DeepHit/Output/weights/Adam_1/readIdentity%DeepHit/DeepHit/Output/weights/Adam_1*
T0*)
_class
loc:@DeepHit/Output/weights
Р
4DeepHit/DeepHit/Output/biases/Adam/Initializer/zerosConst*(
_class
loc:@DeepHit/Output/biases*
dtype0*
valueBЮ*    
Э
"DeepHit/DeepHit/Output/biases/Adam
VariableV2*(
_class
loc:@DeepHit/Output/biases*
	container *
dtype0*
shape:Ю*
shared_name 
й
)DeepHit/DeepHit/Output/biases/Adam/AssignAssign"DeepHit/DeepHit/Output/biases/Adam4DeepHit/DeepHit/Output/biases/Adam/Initializer/zeros*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking(*
validate_shape(
К
'DeepHit/DeepHit/Output/biases/Adam/readIdentity"DeepHit/DeepHit/Output/biases/Adam*
T0*(
_class
loc:@DeepHit/Output/biases
Т
6DeepHit/DeepHit/Output/biases/Adam_1/Initializer/zerosConst*(
_class
loc:@DeepHit/Output/biases*
dtype0*
valueBЮ*    
Я
$DeepHit/DeepHit/Output/biases/Adam_1
VariableV2*(
_class
loc:@DeepHit/Output/biases*
	container *
dtype0*
shape:Ю*
shared_name 
п
+DeepHit/DeepHit/Output/biases/Adam_1/AssignAssign$DeepHit/DeepHit/Output/biases/Adam_16DeepHit/DeepHit/Output/biases/Adam_1/Initializer/zeros*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking(*
validate_shape(
О
)DeepHit/DeepHit/Output/biases/Adam_1/readIdentity$DeepHit/DeepHit/Output/biases/Adam_1*
T0*(
_class
loc:@DeepHit/Output/biases
?
DeepHit/Adam/beta1Const*
dtype0*
valueB
 *fff?
?
DeepHit/Adam/beta2Const*
dtype0*
valueB
 *wЊ?
A
DeepHit/Adam/epsilonConst*
dtype0*
valueB
 *wћ+2
–
=DeepHit/Adam/update_DeepHit/fully_connected/weights/ApplyAdam	ApplyAdamDeepHit/fully_connected/weights,DeepHit/DeepHit/fully_connected/weights/Adam.DeepHit/DeepHit/fully_connected/weights/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonDeepHit/gradients/AddN_9*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights*
use_locking( *
use_nesterov( 
Д
<DeepHit/Adam/update_DeepHit/fully_connected/biases/ApplyAdam	ApplyAdamDeepHit/fully_connected/biases+DeepHit/DeepHit/fully_connected/biases/Adam-DeepHit/DeepHit/fully_connected/biases/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonQDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@DeepHit/fully_connected/biases*
use_locking( *
use_nesterov( 
Џ
?DeepHit/Adam/update_DeepHit/fully_connected_1/weights/ApplyAdam	ApplyAdam!DeepHit/fully_connected_1/weights.DeepHit/DeepHit/fully_connected_1/weights/Adam0DeepHit/DeepHit/fully_connected_1/weights/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonDeepHit/gradients/AddN_8*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
use_locking( *
use_nesterov( 
Р
>DeepHit/Adam/update_DeepHit/fully_connected_1/biases/ApplyAdam	ApplyAdam DeepHit/fully_connected_1/biases-DeepHit/DeepHit/fully_connected_1/biases/Adam/DeepHit/DeepHit/fully_connected_1/biases/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonSDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
use_locking( *
use_nesterov( 
Џ
?DeepHit/Adam/update_DeepHit/fully_connected_2/weights/ApplyAdam	ApplyAdam!DeepHit/fully_connected_2/weights.DeepHit/DeepHit/fully_connected_2/weights/Adam0DeepHit/DeepHit/fully_connected_2/weights/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonDeepHit/gradients/AddN_7*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
use_locking( *
use_nesterov( 
Р
>DeepHit/Adam/update_DeepHit/fully_connected_2/biases/ApplyAdam	ApplyAdam DeepHit/fully_connected_2/biases-DeepHit/DeepHit/fully_connected_2/biases/Adam/DeepHit/DeepHit/fully_connected_2/biases/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonSDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
use_locking( *
use_nesterov( 
Џ
?DeepHit/Adam/update_DeepHit/fully_connected_3/weights/ApplyAdam	ApplyAdam!DeepHit/fully_connected_3/weights.DeepHit/DeepHit/fully_connected_3/weights/Adam0DeepHit/DeepHit/fully_connected_3/weights/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonDeepHit/gradients/AddN_4*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
use_locking( *
use_nesterov( 
Р
>DeepHit/Adam/update_DeepHit/fully_connected_3/biases/ApplyAdam	ApplyAdam DeepHit/fully_connected_3/biases-DeepHit/DeepHit/fully_connected_3/biases/Adam/DeepHit/DeepHit/fully_connected_3/biases/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonSDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
use_locking( *
use_nesterov( 
Џ
?DeepHit/Adam/update_DeepHit/fully_connected_4/weights/ApplyAdam	ApplyAdam!DeepHit/fully_connected_4/weights.DeepHit/DeepHit/fully_connected_4/weights/Adam0DeepHit/DeepHit/fully_connected_4/weights/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonDeepHit/gradients/AddN_6*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
use_locking( *
use_nesterov( 
Р
>DeepHit/Adam/update_DeepHit/fully_connected_4/biases/ApplyAdam	ApplyAdam DeepHit/fully_connected_4/biases-DeepHit/DeepHit/fully_connected_4/biases/Adam/DeepHit/DeepHit/fully_connected_4/biases/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonSDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
use_locking( *
use_nesterov( 
£
4DeepHit/Adam/update_DeepHit/Output/weights/ApplyAdam	ApplyAdamDeepHit/Output/weights#DeepHit/DeepHit/Output/weights/Adam%DeepHit/DeepHit/Output/weights/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonDeepHit/gradients/AddN_3*
T0*)
_class
loc:@DeepHit/Output/weights*
use_locking( *
use_nesterov( 
ќ
3DeepHit/Adam/update_DeepHit/Output/biases/ApplyAdam	ApplyAdamDeepHit/Output/biases"DeepHit/DeepHit/Output/biases/Adam$DeepHit/DeepHit/Output/biases/Adam_1DeepHit/beta1_power/readDeepHit/beta2_power/readDeepHit/learning_rateDeepHit/Adam/beta1DeepHit/Adam/beta2DeepHit/Adam/epsilonHDeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/control_dependency_1*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking( *
use_nesterov( 
р
DeepHit/Adam/mulMulDeepHit/beta1_power/readDeepHit/Adam/beta14^DeepHit/Adam/update_DeepHit/Output/biases/ApplyAdam5^DeepHit/Adam/update_DeepHit/Output/weights/ApplyAdam=^DeepHit/Adam/update_DeepHit/fully_connected/biases/ApplyAdam>^DeepHit/Adam/update_DeepHit/fully_connected/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_1/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_1/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_2/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_2/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_3/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_3/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_4/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_4/weights/ApplyAdam*
T0*(
_class
loc:@DeepHit/Output/biases
†
DeepHit/Adam/AssignAssignDeepHit/beta1_powerDeepHit/Adam/mul*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking( *
validate_shape(
т
DeepHit/Adam/mul_1MulDeepHit/beta2_power/readDeepHit/Adam/beta24^DeepHit/Adam/update_DeepHit/Output/biases/ApplyAdam5^DeepHit/Adam/update_DeepHit/Output/weights/ApplyAdam=^DeepHit/Adam/update_DeepHit/fully_connected/biases/ApplyAdam>^DeepHit/Adam/update_DeepHit/fully_connected/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_1/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_1/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_2/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_2/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_3/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_3/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_4/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_4/weights/ApplyAdam*
T0*(
_class
loc:@DeepHit/Output/biases
§
DeepHit/Adam/Assign_1AssignDeepHit/beta2_powerDeepHit/Adam/mul_1*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking( *
validate_shape(
Ї
DeepHit/AdamNoOp^DeepHit/Adam/Assign^DeepHit/Adam/Assign_14^DeepHit/Adam/update_DeepHit/Output/biases/ApplyAdam5^DeepHit/Adam/update_DeepHit/Output/weights/ApplyAdam=^DeepHit/Adam/update_DeepHit/fully_connected/biases/ApplyAdam>^DeepHit/Adam/update_DeepHit/fully_connected/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_1/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_1/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_2/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_2/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_3/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_3/weights/ApplyAdam?^DeepHit/Adam/update_DeepHit/fully_connected_4/biases/ApplyAdam@^DeepHit/Adam/update_DeepHit/fully_connected_4/weights/ApplyAdam
A
save/filename/inputConst*
dtype0*
valueB Bmodel
V
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: 
M

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: 
ћ
save/SaveV2/tensor_namesConst*
dtype0*Ы
valueСBО&B"DeepHit/DeepHit/Output/biases/AdamB$DeepHit/DeepHit/Output/biases/Adam_1B#DeepHit/DeepHit/Output/weights/AdamB%DeepHit/DeepHit/Output/weights/Adam_1B+DeepHit/DeepHit/fully_connected/biases/AdamB-DeepHit/DeepHit/fully_connected/biases/Adam_1B,DeepHit/DeepHit/fully_connected/weights/AdamB.DeepHit/DeepHit/fully_connected/weights/Adam_1B-DeepHit/DeepHit/fully_connected_1/biases/AdamB/DeepHit/DeepHit/fully_connected_1/biases/Adam_1B.DeepHit/DeepHit/fully_connected_1/weights/AdamB0DeepHit/DeepHit/fully_connected_1/weights/Adam_1B-DeepHit/DeepHit/fully_connected_2/biases/AdamB/DeepHit/DeepHit/fully_connected_2/biases/Adam_1B.DeepHit/DeepHit/fully_connected_2/weights/AdamB0DeepHit/DeepHit/fully_connected_2/weights/Adam_1B-DeepHit/DeepHit/fully_connected_3/biases/AdamB/DeepHit/DeepHit/fully_connected_3/biases/Adam_1B.DeepHit/DeepHit/fully_connected_3/weights/AdamB0DeepHit/DeepHit/fully_connected_3/weights/Adam_1B-DeepHit/DeepHit/fully_connected_4/biases/AdamB/DeepHit/DeepHit/fully_connected_4/biases/Adam_1B.DeepHit/DeepHit/fully_connected_4/weights/AdamB0DeepHit/DeepHit/fully_connected_4/weights/Adam_1BDeepHit/Output/biasesBDeepHit/Output/weightsBDeepHit/beta1_powerBDeepHit/beta2_powerBDeepHit/fully_connected/biasesBDeepHit/fully_connected/weightsB DeepHit/fully_connected_1/biasesB!DeepHit/fully_connected_1/weightsB DeepHit/fully_connected_2/biasesB!DeepHit/fully_connected_2/weightsB DeepHit/fully_connected_3/biasesB!DeepHit/fully_connected_3/weightsB DeepHit/fully_connected_4/biasesB!DeepHit/fully_connected_4/weights
У
save/SaveV2/shape_and_slicesConst*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Х
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"DeepHit/DeepHit/Output/biases/Adam$DeepHit/DeepHit/Output/biases/Adam_1#DeepHit/DeepHit/Output/weights/Adam%DeepHit/DeepHit/Output/weights/Adam_1+DeepHit/DeepHit/fully_connected/biases/Adam-DeepHit/DeepHit/fully_connected/biases/Adam_1,DeepHit/DeepHit/fully_connected/weights/Adam.DeepHit/DeepHit/fully_connected/weights/Adam_1-DeepHit/DeepHit/fully_connected_1/biases/Adam/DeepHit/DeepHit/fully_connected_1/biases/Adam_1.DeepHit/DeepHit/fully_connected_1/weights/Adam0DeepHit/DeepHit/fully_connected_1/weights/Adam_1-DeepHit/DeepHit/fully_connected_2/biases/Adam/DeepHit/DeepHit/fully_connected_2/biases/Adam_1.DeepHit/DeepHit/fully_connected_2/weights/Adam0DeepHit/DeepHit/fully_connected_2/weights/Adam_1-DeepHit/DeepHit/fully_connected_3/biases/Adam/DeepHit/DeepHit/fully_connected_3/biases/Adam_1.DeepHit/DeepHit/fully_connected_3/weights/Adam0DeepHit/DeepHit/fully_connected_3/weights/Adam_1-DeepHit/DeepHit/fully_connected_4/biases/Adam/DeepHit/DeepHit/fully_connected_4/biases/Adam_1.DeepHit/DeepHit/fully_connected_4/weights/Adam0DeepHit/DeepHit/fully_connected_4/weights/Adam_1DeepHit/Output/biasesDeepHit/Output/weightsDeepHit/beta1_powerDeepHit/beta2_powerDeepHit/fully_connected/biasesDeepHit/fully_connected/weights DeepHit/fully_connected_1/biases!DeepHit/fully_connected_1/weights DeepHit/fully_connected_2/biases!DeepHit/fully_connected_2/weights DeepHit/fully_connected_3/biases!DeepHit/fully_connected_3/weights DeepHit/fully_connected_4/biases!DeepHit/fully_connected_4/weights*4
dtypes*
(2&
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
ё
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*Ы
valueСBО&B"DeepHit/DeepHit/Output/biases/AdamB$DeepHit/DeepHit/Output/biases/Adam_1B#DeepHit/DeepHit/Output/weights/AdamB%DeepHit/DeepHit/Output/weights/Adam_1B+DeepHit/DeepHit/fully_connected/biases/AdamB-DeepHit/DeepHit/fully_connected/biases/Adam_1B,DeepHit/DeepHit/fully_connected/weights/AdamB.DeepHit/DeepHit/fully_connected/weights/Adam_1B-DeepHit/DeepHit/fully_connected_1/biases/AdamB/DeepHit/DeepHit/fully_connected_1/biases/Adam_1B.DeepHit/DeepHit/fully_connected_1/weights/AdamB0DeepHit/DeepHit/fully_connected_1/weights/Adam_1B-DeepHit/DeepHit/fully_connected_2/biases/AdamB/DeepHit/DeepHit/fully_connected_2/biases/Adam_1B.DeepHit/DeepHit/fully_connected_2/weights/AdamB0DeepHit/DeepHit/fully_connected_2/weights/Adam_1B-DeepHit/DeepHit/fully_connected_3/biases/AdamB/DeepHit/DeepHit/fully_connected_3/biases/Adam_1B.DeepHit/DeepHit/fully_connected_3/weights/AdamB0DeepHit/DeepHit/fully_connected_3/weights/Adam_1B-DeepHit/DeepHit/fully_connected_4/biases/AdamB/DeepHit/DeepHit/fully_connected_4/biases/Adam_1B.DeepHit/DeepHit/fully_connected_4/weights/AdamB0DeepHit/DeepHit/fully_connected_4/weights/Adam_1BDeepHit/Output/biasesBDeepHit/Output/weightsBDeepHit/beta1_powerBDeepHit/beta2_powerBDeepHit/fully_connected/biasesBDeepHit/fully_connected/weightsB DeepHit/fully_connected_1/biasesB!DeepHit/fully_connected_1/weightsB DeepHit/fully_connected_2/biasesB!DeepHit/fully_connected_2/weightsB DeepHit/fully_connected_3/biasesB!DeepHit/fully_connected_3/weightsB DeepHit/fully_connected_4/biasesB!DeepHit/fully_connected_4/weights
•
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
™
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*4
dtypes*
(2&
•
save/AssignAssign"DeepHit/DeepHit/Output/biases/Adamsave/RestoreV2*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking(*
validate_shape(
Ђ
save/Assign_1Assign$DeepHit/DeepHit/Output/biases/Adam_1save/RestoreV2:1*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking(*
validate_shape(
Ђ
save/Assign_2Assign#DeepHit/DeepHit/Output/weights/Adamsave/RestoreV2:2*
T0*)
_class
loc:@DeepHit/Output/weights*
use_locking(*
validate_shape(
≠
save/Assign_3Assign%DeepHit/DeepHit/Output/weights/Adam_1save/RestoreV2:3*
T0*)
_class
loc:@DeepHit/Output/weights*
use_locking(*
validate_shape(
ї
save/Assign_4Assign+DeepHit/DeepHit/fully_connected/biases/Adamsave/RestoreV2:4*
T0*1
_class'
%#loc:@DeepHit/fully_connected/biases*
use_locking(*
validate_shape(
љ
save/Assign_5Assign-DeepHit/DeepHit/fully_connected/biases/Adam_1save/RestoreV2:5*
T0*1
_class'
%#loc:@DeepHit/fully_connected/biases*
use_locking(*
validate_shape(
љ
save/Assign_6Assign,DeepHit/DeepHit/fully_connected/weights/Adamsave/RestoreV2:6*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights*
use_locking(*
validate_shape(
њ
save/Assign_7Assign.DeepHit/DeepHit/fully_connected/weights/Adam_1save/RestoreV2:7*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights*
use_locking(*
validate_shape(
њ
save/Assign_8Assign-DeepHit/DeepHit/fully_connected_1/biases/Adamsave/RestoreV2:8*
T0*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
use_locking(*
validate_shape(
Ѕ
save/Assign_9Assign/DeepHit/DeepHit/fully_connected_1/biases/Adam_1save/RestoreV2:9*
T0*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
use_locking(*
validate_shape(
√
save/Assign_10Assign.DeepHit/DeepHit/fully_connected_1/weights/Adamsave/RestoreV2:10*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
use_locking(*
validate_shape(
≈
save/Assign_11Assign0DeepHit/DeepHit/fully_connected_1/weights/Adam_1save/RestoreV2:11*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
use_locking(*
validate_shape(
Ѕ
save/Assign_12Assign-DeepHit/DeepHit/fully_connected_2/biases/Adamsave/RestoreV2:12*
T0*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
use_locking(*
validate_shape(
√
save/Assign_13Assign/DeepHit/DeepHit/fully_connected_2/biases/Adam_1save/RestoreV2:13*
T0*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
use_locking(*
validate_shape(
√
save/Assign_14Assign.DeepHit/DeepHit/fully_connected_2/weights/Adamsave/RestoreV2:14*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
use_locking(*
validate_shape(
≈
save/Assign_15Assign0DeepHit/DeepHit/fully_connected_2/weights/Adam_1save/RestoreV2:15*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
use_locking(*
validate_shape(
Ѕ
save/Assign_16Assign-DeepHit/DeepHit/fully_connected_3/biases/Adamsave/RestoreV2:16*
T0*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
use_locking(*
validate_shape(
√
save/Assign_17Assign/DeepHit/DeepHit/fully_connected_3/biases/Adam_1save/RestoreV2:17*
T0*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
use_locking(*
validate_shape(
√
save/Assign_18Assign.DeepHit/DeepHit/fully_connected_3/weights/Adamsave/RestoreV2:18*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
use_locking(*
validate_shape(
≈
save/Assign_19Assign0DeepHit/DeepHit/fully_connected_3/weights/Adam_1save/RestoreV2:19*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
use_locking(*
validate_shape(
Ѕ
save/Assign_20Assign-DeepHit/DeepHit/fully_connected_4/biases/Adamsave/RestoreV2:20*
T0*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
use_locking(*
validate_shape(
√
save/Assign_21Assign/DeepHit/DeepHit/fully_connected_4/biases/Adam_1save/RestoreV2:21*
T0*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
use_locking(*
validate_shape(
√
save/Assign_22Assign.DeepHit/DeepHit/fully_connected_4/weights/Adamsave/RestoreV2:22*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
use_locking(*
validate_shape(
≈
save/Assign_23Assign0DeepHit/DeepHit/fully_connected_4/weights/Adam_1save/RestoreV2:23*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
use_locking(*
validate_shape(
Ю
save/Assign_24AssignDeepHit/Output/biasessave/RestoreV2:24*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking(*
validate_shape(
†
save/Assign_25AssignDeepHit/Output/weightssave/RestoreV2:25*
T0*)
_class
loc:@DeepHit/Output/weights*
use_locking(*
validate_shape(
Ь
save/Assign_26AssignDeepHit/beta1_powersave/RestoreV2:26*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking(*
validate_shape(
Ь
save/Assign_27AssignDeepHit/beta2_powersave/RestoreV2:27*
T0*(
_class
loc:@DeepHit/Output/biases*
use_locking(*
validate_shape(
∞
save/Assign_28AssignDeepHit/fully_connected/biasessave/RestoreV2:28*
T0*1
_class'
%#loc:@DeepHit/fully_connected/biases*
use_locking(*
validate_shape(
≤
save/Assign_29AssignDeepHit/fully_connected/weightssave/RestoreV2:29*
T0*2
_class(
&$loc:@DeepHit/fully_connected/weights*
use_locking(*
validate_shape(
і
save/Assign_30Assign DeepHit/fully_connected_1/biasessave/RestoreV2:30*
T0*3
_class)
'%loc:@DeepHit/fully_connected_1/biases*
use_locking(*
validate_shape(
ґ
save/Assign_31Assign!DeepHit/fully_connected_1/weightssave/RestoreV2:31*
T0*4
_class*
(&loc:@DeepHit/fully_connected_1/weights*
use_locking(*
validate_shape(
і
save/Assign_32Assign DeepHit/fully_connected_2/biasessave/RestoreV2:32*
T0*3
_class)
'%loc:@DeepHit/fully_connected_2/biases*
use_locking(*
validate_shape(
ґ
save/Assign_33Assign!DeepHit/fully_connected_2/weightssave/RestoreV2:33*
T0*4
_class*
(&loc:@DeepHit/fully_connected_2/weights*
use_locking(*
validate_shape(
і
save/Assign_34Assign DeepHit/fully_connected_3/biasessave/RestoreV2:34*
T0*3
_class)
'%loc:@DeepHit/fully_connected_3/biases*
use_locking(*
validate_shape(
ґ
save/Assign_35Assign!DeepHit/fully_connected_3/weightssave/RestoreV2:35*
T0*4
_class*
(&loc:@DeepHit/fully_connected_3/weights*
use_locking(*
validate_shape(
і
save/Assign_36Assign DeepHit/fully_connected_4/biasessave/RestoreV2:36*
T0*3
_class)
'%loc:@DeepHit/fully_connected_4/biases*
use_locking(*
validate_shape(
ґ
save/Assign_37Assign!DeepHit/fully_connected_4/weightssave/RestoreV2:37*
T0*4
_class*
(&loc:@DeepHit/fully_connected_4/weights*
use_locking(*
validate_shape(
Т
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
¬
initNoOp*^DeepHit/DeepHit/Output/biases/Adam/Assign,^DeepHit/DeepHit/Output/biases/Adam_1/Assign+^DeepHit/DeepHit/Output/weights/Adam/Assign-^DeepHit/DeepHit/Output/weights/Adam_1/Assign3^DeepHit/DeepHit/fully_connected/biases/Adam/Assign5^DeepHit/DeepHit/fully_connected/biases/Adam_1/Assign4^DeepHit/DeepHit/fully_connected/weights/Adam/Assign6^DeepHit/DeepHit/fully_connected/weights/Adam_1/Assign5^DeepHit/DeepHit/fully_connected_1/biases/Adam/Assign7^DeepHit/DeepHit/fully_connected_1/biases/Adam_1/Assign6^DeepHit/DeepHit/fully_connected_1/weights/Adam/Assign8^DeepHit/DeepHit/fully_connected_1/weights/Adam_1/Assign5^DeepHit/DeepHit/fully_connected_2/biases/Adam/Assign7^DeepHit/DeepHit/fully_connected_2/biases/Adam_1/Assign6^DeepHit/DeepHit/fully_connected_2/weights/Adam/Assign8^DeepHit/DeepHit/fully_connected_2/weights/Adam_1/Assign5^DeepHit/DeepHit/fully_connected_3/biases/Adam/Assign7^DeepHit/DeepHit/fully_connected_3/biases/Adam_1/Assign6^DeepHit/DeepHit/fully_connected_3/weights/Adam/Assign8^DeepHit/DeepHit/fully_connected_3/weights/Adam_1/Assign5^DeepHit/DeepHit/fully_connected_4/biases/Adam/Assign7^DeepHit/DeepHit/fully_connected_4/biases/Adam_1/Assign6^DeepHit/DeepHit/fully_connected_4/weights/Adam/Assign8^DeepHit/DeepHit/fully_connected_4/weights/Adam_1/Assign^DeepHit/Output/biases/Assign^DeepHit/Output/weights/Assign^DeepHit/beta1_power/Assign^DeepHit/beta2_power/Assign&^DeepHit/fully_connected/biases/Assign'^DeepHit/fully_connected/weights/Assign(^DeepHit/fully_connected_1/biases/Assign)^DeepHit/fully_connected_1/weights/Assign(^DeepHit/fully_connected_2/biases/Assign)^DeepHit/fully_connected_2/weights/Assign(^DeepHit/fully_connected_3/biases/Assign)^DeepHit/fully_connected_3/weights/Assign(^DeepHit/fully_connected_4/biases/Assign)^DeepHit/fully_connected_4/weights/Assign
=
DeepHit/batch_size_1Placeholder*
dtype0*
shape: 
@
DeepHit/learning_rate_1Placeholder*
dtype0*
shape: 
C
DeepHit/keep_probability_1Placeholder*
dtype0*
shape: 
8
DeepHit/alpha_1Placeholder*
dtype0*
shape: 
7
DeepHit/beta_1Placeholder*
dtype0*
shape: 
8
DeepHit/gamma_1Placeholder*
dtype0*
shape: 
J
DeepHit/inputs_1Placeholder*
dtype0*
shape:€€€€€€€€€`
J
DeepHit/labels_1Placeholder*
dtype0*
shape:€€€€€€€€€
P
DeepHit/timetoevents_1Placeholder*
dtype0*
shape:€€€€€€€€€
N
DeepHit/mask1_1Placeholder*
dtype0*!
shape:€€€€€€€€€П
J
DeepHit/mask2_1Placeholder*
dtype0*
shape:€€€€€€€€€П
Ђ
@DeepHit/fully_connected/weights/Initializer/random_uniform/shapeConst*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
dtype0*
valueB"`   
   
°
>DeepHit/fully_connected/weights/Initializer/random_uniform/minConst*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
dtype0*
valueB
 *†sЊ
°
>DeepHit/fully_connected/weights/Initializer/random_uniform/maxConst*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
dtype0*
valueB
 *†s>
А
HDeepHit/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniform@DeepHit/fully_connected/weights/Initializer/random_uniform/shape*
T0*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
dtype0*

seed *
seed2 
Д
>DeepHit/fully_connected/weights/Initializer/random_uniform/subSub>DeepHit/fully_connected/weights/Initializer/random_uniform/max>DeepHit/fully_connected/weights/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@DeepHit/fully_connected/weights_1
О
>DeepHit/fully_connected/weights/Initializer/random_uniform/mulMulHDeepHit/fully_connected/weights/Initializer/random_uniform/RandomUniform>DeepHit/fully_connected/weights/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@DeepHit/fully_connected/weights_1
В
:DeepHit/fully_connected/weights/Initializer/random_uniformAddV2>DeepHit/fully_connected/weights/Initializer/random_uniform/mul>DeepHit/fully_connected/weights/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@DeepHit/fully_connected/weights_1
в
!DeepHit/fully_connected/weights_1VarHandleOp*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:`
*0
shared_name!DeepHit/fully_connected/weights
y
@DeepHit/fully_connected/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp!DeepHit/fully_connected/weights_1
Њ
(DeepHit/fully_connected/weights/Assign_1AssignVariableOp!DeepHit/fully_connected/weights_1:DeepHit/fully_connected/weights/Initializer/random_uniform*
dtype0*
validate_shape( 
u
3DeepHit/fully_connected/weights/Read/ReadVariableOpReadVariableOp!DeepHit/fully_connected/weights_1*
dtype0
В
@DeepHit/fully_connected/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!DeepHit/fully_connected/weights_1*
dtype0
И
3DeepHit/fully_connected/kernel/Regularizer/Square_1Square@DeepHit/fully_connected/kernel/Regularizer/Square/ReadVariableOp*
T0
g
2DeepHit/fully_connected/kernel/Regularizer/Const_1Const*
dtype0*
valueB"       
∆
0DeepHit/fully_connected/kernel/Regularizer/Sum_1Sum3DeepHit/fully_connected/kernel/Regularizer/Square_12DeepHit/fully_connected/kernel/Regularizer/Const_1*
T0*

Tidx0*
	keep_dims( 
_
2DeepHit/fully_connected/kernel/Regularizer/mul/x_1Const*
dtype0*
valueB
 *ЈQ8
¶
0DeepHit/fully_connected/kernel/Regularizer/mul_1Mul2DeepHit/fully_connected/kernel/Regularizer/mul/x_10DeepHit/fully_connected/kernel/Regularizer/Sum_1*
T0
Ш
2DeepHit/fully_connected/biases/Initializer/zeros_1Const*3
_class)
'%loc:@DeepHit/fully_connected/biases_1*
dtype0*
valueB
*    
џ
 DeepHit/fully_connected/biases_1VarHandleOp*3
_class)
'%loc:@DeepHit/fully_connected/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:
*/
shared_name DeepHit/fully_connected/biases
w
?DeepHit/fully_connected/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOp DeepHit/fully_connected/biases_1
і
'DeepHit/fully_connected/biases/Assign_1AssignVariableOp DeepHit/fully_connected/biases_12DeepHit/fully_connected/biases/Initializer/zeros_1*
dtype0*
validate_shape( 
s
2DeepHit/fully_connected/biases/Read/ReadVariableOpReadVariableOp DeepHit/fully_connected/biases_1*
dtype0
o
-DeepHit/fully_connected/MatMul/ReadVariableOpReadVariableOp!DeepHit/fully_connected/weights_1*
dtype0
Ъ
 DeepHit/fully_connected/MatMul_1MatMulDeepHit/inputs_1-DeepHit/fully_connected/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b( 
o
.DeepHit/fully_connected/BiasAdd/ReadVariableOpReadVariableOp DeepHit/fully_connected/biases_1*
dtype0
Ю
!DeepHit/fully_connected/BiasAdd_1BiasAdd DeepHit/fully_connected/MatMul_1.DeepHit/fully_connected/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
P
DeepHit/fully_connected/Elu_1Elu!DeepHit/fully_connected/BiasAdd_1*
T0
<
DeepHit/sub/x_1Const*
dtype0*
valueB
 *  А?
K
DeepHit/sub_10SubDeepHit/sub/x_1DeepHit/keep_probability_1*
T0
D
DeepHit/dropout/Const_1Const*
dtype0*
valueB
 *  А?
N
DeepHit/dropout/Sub_1SubDeepHit/dropout/Const_1DeepHit/sub_10*
T0
c
DeepHit/dropout/RealDiv_1RealDivDeepHit/fully_connected/Elu_1DeepHit/dropout/Sub_1*
T0
X
DeepHit/dropout/Shape_1ShapeDeepHit/fully_connected/Elu_1*
T0*
out_type0
З
.DeepHit/dropout/random_uniform/RandomUniform_1RandomUniformDeepHit/dropout/Shape_1*
T0*
dtype0*

seed *
seed2 
w
DeepHit/dropout/GreaterEqual_1GreaterEqual.DeepHit/dropout/random_uniform/RandomUniform_1DeepHit/sub_10*
T0
f
DeepHit/dropout/Cast_1CastDeepHit/dropout/GreaterEqual_1*

DstT0*

SrcT0
*
Truncate( 
X
DeepHit/dropout/Mul_1MulDeepHit/dropout/RealDiv_1DeepHit/dropout/Cast_1*
T0
ѓ
BDeepHit/fully_connected_1/weights/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
dtype0*
valueB"
   
   
•
@DeepHit/fully_connected_1/weights/Initializer/random_uniform/minConst*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
dtype0*
valueB
 *М7њ
•
@DeepHit/fully_connected_1/weights/Initializer/random_uniform/maxConst*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
dtype0*
valueB
 *М7?
Ж
JDeepHit/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformBDeepHit/fully_connected_1/weights/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
dtype0*

seed *
seed2 
М
@DeepHit/fully_connected_1/weights/Initializer/random_uniform/subSub@DeepHit/fully_connected_1/weights/Initializer/random_uniform/max@DeepHit/fully_connected_1/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1
Ц
@DeepHit/fully_connected_1/weights/Initializer/random_uniform/mulMulJDeepHit/fully_connected_1/weights/Initializer/random_uniform/RandomUniform@DeepHit/fully_connected_1/weights/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1
К
<DeepHit/fully_connected_1/weights/Initializer/random_uniformAddV2@DeepHit/fully_connected_1/weights/Initializer/random_uniform/mul@DeepHit/fully_connected_1/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1
и
#DeepHit/fully_connected_1/weights_1VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:

*2
shared_name#!DeepHit/fully_connected_1/weights
}
BDeepHit/fully_connected_1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp#DeepHit/fully_connected_1/weights_1
ƒ
*DeepHit/fully_connected_1/weights/Assign_1AssignVariableOp#DeepHit/fully_connected_1/weights_1<DeepHit/fully_connected_1/weights/Initializer/random_uniform*
dtype0*
validate_shape( 
y
5DeepHit/fully_connected_1/weights/Read/ReadVariableOpReadVariableOp#DeepHit/fully_connected_1/weights_1*
dtype0
Ж
BDeepHit/fully_connected_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#DeepHit/fully_connected_1/weights_1*
dtype0
М
5DeepHit/fully_connected_1/kernel/Regularizer/Square_1SquareBDeepHit/fully_connected_1/kernel/Regularizer/Square/ReadVariableOp*
T0
i
4DeepHit/fully_connected_1/kernel/Regularizer/Const_1Const*
dtype0*
valueB"       
ћ
2DeepHit/fully_connected_1/kernel/Regularizer/Sum_1Sum5DeepHit/fully_connected_1/kernel/Regularizer/Square_14DeepHit/fully_connected_1/kernel/Regularizer/Const_1*
T0*

Tidx0*
	keep_dims( 
a
4DeepHit/fully_connected_1/kernel/Regularizer/mul/x_1Const*
dtype0*
valueB
 *ЈQ8
ђ
2DeepHit/fully_connected_1/kernel/Regularizer/mul_1Mul4DeepHit/fully_connected_1/kernel/Regularizer/mul/x_12DeepHit/fully_connected_1/kernel/Regularizer/Sum_1*
T0
Ь
4DeepHit/fully_connected_1/biases/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1*
dtype0*
valueB
*    
б
"DeepHit/fully_connected_1/biases_1VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:
*1
shared_name" DeepHit/fully_connected_1/biases
{
ADeepHit/fully_connected_1/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOp"DeepHit/fully_connected_1/biases_1
Ї
)DeepHit/fully_connected_1/biases/Assign_1AssignVariableOp"DeepHit/fully_connected_1/biases_14DeepHit/fully_connected_1/biases/Initializer/zeros_1*
dtype0*
validate_shape( 
w
4DeepHit/fully_connected_1/biases/Read/ReadVariableOpReadVariableOp"DeepHit/fully_connected_1/biases_1*
dtype0
s
/DeepHit/fully_connected_1/MatMul/ReadVariableOpReadVariableOp#DeepHit/fully_connected_1/weights_1*
dtype0
£
"DeepHit/fully_connected_1/MatMul_1MatMulDeepHit/dropout/Mul_1/DeepHit/fully_connected_1/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b( 
s
0DeepHit/fully_connected_1/BiasAdd/ReadVariableOpReadVariableOp"DeepHit/fully_connected_1/biases_1*
dtype0
§
#DeepHit/fully_connected_1/BiasAdd_1BiasAdd"DeepHit/fully_connected_1/MatMul_10DeepHit/fully_connected_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
T
DeepHit/fully_connected_1/Elu_1Elu#DeepHit/fully_connected_1/BiasAdd_1*
T0
>
DeepHit/sub_1/x_1Const*
dtype0*
valueB
 *  А?
N
DeepHit/sub_1_1SubDeepHit/sub_1/x_1DeepHit/keep_probability_1*
T0
F
DeepHit/dropout_1/Const_1Const*
dtype0*
valueB
 *  А?
S
DeepHit/dropout_1/Sub_1SubDeepHit/dropout_1/Const_1DeepHit/sub_1_1*
T0
i
DeepHit/dropout_1/RealDiv_1RealDivDeepHit/fully_connected_1/Elu_1DeepHit/dropout_1/Sub_1*
T0
\
DeepHit/dropout_1/Shape_1ShapeDeepHit/fully_connected_1/Elu_1*
T0*
out_type0
Л
0DeepHit/dropout_1/random_uniform/RandomUniform_1RandomUniformDeepHit/dropout_1/Shape_1*
T0*
dtype0*

seed *
seed2 
|
 DeepHit/dropout_1/GreaterEqual_1GreaterEqual0DeepHit/dropout_1/random_uniform/RandomUniform_1DeepHit/sub_1_1*
T0
j
DeepHit/dropout_1/Cast_1Cast DeepHit/dropout_1/GreaterEqual_1*

DstT0*

SrcT0
*
Truncate( 
^
DeepHit/dropout_1/Mul_1MulDeepHit/dropout_1/RealDiv_1DeepHit/dropout_1/Cast_1*
T0
ѓ
BDeepHit/fully_connected_2/weights/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
dtype0*
valueB"
   
   
•
@DeepHit/fully_connected_2/weights/Initializer/random_uniform/minConst*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
dtype0*
valueB
 *М7њ
•
@DeepHit/fully_connected_2/weights/Initializer/random_uniform/maxConst*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
dtype0*
valueB
 *М7?
Ж
JDeepHit/fully_connected_2/weights/Initializer/random_uniform/RandomUniformRandomUniformBDeepHit/fully_connected_2/weights/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
dtype0*

seed *
seed2 
М
@DeepHit/fully_connected_2/weights/Initializer/random_uniform/subSub@DeepHit/fully_connected_2/weights/Initializer/random_uniform/max@DeepHit/fully_connected_2/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1
Ц
@DeepHit/fully_connected_2/weights/Initializer/random_uniform/mulMulJDeepHit/fully_connected_2/weights/Initializer/random_uniform/RandomUniform@DeepHit/fully_connected_2/weights/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1
К
<DeepHit/fully_connected_2/weights/Initializer/random_uniformAddV2@DeepHit/fully_connected_2/weights/Initializer/random_uniform/mul@DeepHit/fully_connected_2/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1
и
#DeepHit/fully_connected_2/weights_1VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:

*2
shared_name#!DeepHit/fully_connected_2/weights
}
BDeepHit/fully_connected_2/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp#DeepHit/fully_connected_2/weights_1
ƒ
*DeepHit/fully_connected_2/weights/Assign_1AssignVariableOp#DeepHit/fully_connected_2/weights_1<DeepHit/fully_connected_2/weights/Initializer/random_uniform*
dtype0*
validate_shape( 
y
5DeepHit/fully_connected_2/weights/Read/ReadVariableOpReadVariableOp#DeepHit/fully_connected_2/weights_1*
dtype0
Ж
BDeepHit/fully_connected_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#DeepHit/fully_connected_2/weights_1*
dtype0
М
5DeepHit/fully_connected_2/kernel/Regularizer/Square_1SquareBDeepHit/fully_connected_2/kernel/Regularizer/Square/ReadVariableOp*
T0
i
4DeepHit/fully_connected_2/kernel/Regularizer/Const_1Const*
dtype0*
valueB"       
ћ
2DeepHit/fully_connected_2/kernel/Regularizer/Sum_1Sum5DeepHit/fully_connected_2/kernel/Regularizer/Square_14DeepHit/fully_connected_2/kernel/Regularizer/Const_1*
T0*

Tidx0*
	keep_dims( 
a
4DeepHit/fully_connected_2/kernel/Regularizer/mul/x_1Const*
dtype0*
valueB
 *ЈQ8
ђ
2DeepHit/fully_connected_2/kernel/Regularizer/mul_1Mul4DeepHit/fully_connected_2/kernel/Regularizer/mul/x_12DeepHit/fully_connected_2/kernel/Regularizer/Sum_1*
T0
Ь
4DeepHit/fully_connected_2/biases/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1*
dtype0*
valueB
*    
б
"DeepHit/fully_connected_2/biases_1VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:
*1
shared_name" DeepHit/fully_connected_2/biases
{
ADeepHit/fully_connected_2/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOp"DeepHit/fully_connected_2/biases_1
Ї
)DeepHit/fully_connected_2/biases/Assign_1AssignVariableOp"DeepHit/fully_connected_2/biases_14DeepHit/fully_connected_2/biases/Initializer/zeros_1*
dtype0*
validate_shape( 
w
4DeepHit/fully_connected_2/biases/Read/ReadVariableOpReadVariableOp"DeepHit/fully_connected_2/biases_1*
dtype0
s
/DeepHit/fully_connected_2/MatMul/ReadVariableOpReadVariableOp#DeepHit/fully_connected_2/weights_1*
dtype0
•
"DeepHit/fully_connected_2/MatMul_1MatMulDeepHit/dropout_1/Mul_1/DeepHit/fully_connected_2/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b( 
s
0DeepHit/fully_connected_2/BiasAdd/ReadVariableOpReadVariableOp"DeepHit/fully_connected_2/biases_1*
dtype0
§
#DeepHit/fully_connected_2/BiasAdd_1BiasAdd"DeepHit/fully_connected_2/MatMul_10DeepHit/fully_connected_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
T
DeepHit/fully_connected_2/Elu_1Elu#DeepHit/fully_connected_2/BiasAdd_1*
T0
?
DeepHit/concat/axis_1Const*
dtype0*
value	B :
Д
DeepHit/concat_1ConcatV2DeepHit/inputs_1DeepHit/fully_connected_2/Elu_1DeepHit/concat/axis_1*
N*
T0*

Tidx0
ѓ
BDeepHit/fully_connected_3/weights/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
dtype0*
valueB"j      
•
@DeepHit/fully_connected_3/weights/Initializer/random_uniform/minConst*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
dtype0*
valueB
 *Гt_Њ
•
@DeepHit/fully_connected_3/weights/Initializer/random_uniform/maxConst*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
dtype0*
valueB
 *Гt_>
Ж
JDeepHit/fully_connected_3/weights/Initializer/random_uniform/RandomUniformRandomUniformBDeepHit/fully_connected_3/weights/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
dtype0*

seed *
seed2 
М
@DeepHit/fully_connected_3/weights/Initializer/random_uniform/subSub@DeepHit/fully_connected_3/weights/Initializer/random_uniform/max@DeepHit/fully_connected_3/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1
Ц
@DeepHit/fully_connected_3/weights/Initializer/random_uniform/mulMulJDeepHit/fully_connected_3/weights/Initializer/random_uniform/RandomUniform@DeepHit/fully_connected_3/weights/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1
К
<DeepHit/fully_connected_3/weights/Initializer/random_uniformAddV2@DeepHit/fully_connected_3/weights/Initializer/random_uniform/mul@DeepHit/fully_connected_3/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1
и
#DeepHit/fully_connected_3/weights_1VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:j*2
shared_name#!DeepHit/fully_connected_3/weights
}
BDeepHit/fully_connected_3/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp#DeepHit/fully_connected_3/weights_1
ƒ
*DeepHit/fully_connected_3/weights/Assign_1AssignVariableOp#DeepHit/fully_connected_3/weights_1<DeepHit/fully_connected_3/weights/Initializer/random_uniform*
dtype0*
validate_shape( 
y
5DeepHit/fully_connected_3/weights/Read/ReadVariableOpReadVariableOp#DeepHit/fully_connected_3/weights_1*
dtype0
Ж
BDeepHit/fully_connected_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#DeepHit/fully_connected_3/weights_1*
dtype0
М
5DeepHit/fully_connected_3/kernel/Regularizer/Square_1SquareBDeepHit/fully_connected_3/kernel/Regularizer/Square/ReadVariableOp*
T0
i
4DeepHit/fully_connected_3/kernel/Regularizer/Const_1Const*
dtype0*
valueB"       
ћ
2DeepHit/fully_connected_3/kernel/Regularizer/Sum_1Sum5DeepHit/fully_connected_3/kernel/Regularizer/Square_14DeepHit/fully_connected_3/kernel/Regularizer/Const_1*
T0*

Tidx0*
	keep_dims( 
a
4DeepHit/fully_connected_3/kernel/Regularizer/mul/x_1Const*
dtype0*
valueB
 *ЈQ8
ђ
2DeepHit/fully_connected_3/kernel/Regularizer/mul_1Mul4DeepHit/fully_connected_3/kernel/Regularizer/mul/x_12DeepHit/fully_connected_3/kernel/Regularizer/Sum_1*
T0
Ь
4DeepHit/fully_connected_3/biases/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1*
dtype0*
valueB*    
б
"DeepHit/fully_connected_3/biases_1VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:*1
shared_name" DeepHit/fully_connected_3/biases
{
ADeepHit/fully_connected_3/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOp"DeepHit/fully_connected_3/biases_1
Ї
)DeepHit/fully_connected_3/biases/Assign_1AssignVariableOp"DeepHit/fully_connected_3/biases_14DeepHit/fully_connected_3/biases/Initializer/zeros_1*
dtype0*
validate_shape( 
w
4DeepHit/fully_connected_3/biases/Read/ReadVariableOpReadVariableOp"DeepHit/fully_connected_3/biases_1*
dtype0
s
/DeepHit/fully_connected_3/MatMul/ReadVariableOpReadVariableOp#DeepHit/fully_connected_3/weights_1*
dtype0
Ю
"DeepHit/fully_connected_3/MatMul_1MatMulDeepHit/concat_1/DeepHit/fully_connected_3/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b( 
s
0DeepHit/fully_connected_3/BiasAdd/ReadVariableOpReadVariableOp"DeepHit/fully_connected_3/biases_1*
dtype0
§
#DeepHit/fully_connected_3/BiasAdd_1BiasAdd"DeepHit/fully_connected_3/MatMul_10DeepHit/fully_connected_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
T
DeepHit/fully_connected_3/Elu_1Elu#DeepHit/fully_connected_3/BiasAdd_1*
T0
ѓ
BDeepHit/fully_connected_4/weights/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
dtype0*
valueB"j      
•
@DeepHit/fully_connected_4/weights/Initializer/random_uniform/minConst*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
dtype0*
valueB
 *Гt_Њ
•
@DeepHit/fully_connected_4/weights/Initializer/random_uniform/maxConst*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
dtype0*
valueB
 *Гt_>
Ж
JDeepHit/fully_connected_4/weights/Initializer/random_uniform/RandomUniformRandomUniformBDeepHit/fully_connected_4/weights/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
dtype0*

seed *
seed2 
М
@DeepHit/fully_connected_4/weights/Initializer/random_uniform/subSub@DeepHit/fully_connected_4/weights/Initializer/random_uniform/max@DeepHit/fully_connected_4/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1
Ц
@DeepHit/fully_connected_4/weights/Initializer/random_uniform/mulMulJDeepHit/fully_connected_4/weights/Initializer/random_uniform/RandomUniform@DeepHit/fully_connected_4/weights/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1
К
<DeepHit/fully_connected_4/weights/Initializer/random_uniformAddV2@DeepHit/fully_connected_4/weights/Initializer/random_uniform/mul@DeepHit/fully_connected_4/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1
и
#DeepHit/fully_connected_4/weights_1VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:j*2
shared_name#!DeepHit/fully_connected_4/weights
}
BDeepHit/fully_connected_4/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp#DeepHit/fully_connected_4/weights_1
ƒ
*DeepHit/fully_connected_4/weights/Assign_1AssignVariableOp#DeepHit/fully_connected_4/weights_1<DeepHit/fully_connected_4/weights/Initializer/random_uniform*
dtype0*
validate_shape( 
y
5DeepHit/fully_connected_4/weights/Read/ReadVariableOpReadVariableOp#DeepHit/fully_connected_4/weights_1*
dtype0
Ж
BDeepHit/fully_connected_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#DeepHit/fully_connected_4/weights_1*
dtype0
М
5DeepHit/fully_connected_4/kernel/Regularizer/Square_1SquareBDeepHit/fully_connected_4/kernel/Regularizer/Square/ReadVariableOp*
T0
i
4DeepHit/fully_connected_4/kernel/Regularizer/Const_1Const*
dtype0*
valueB"       
ћ
2DeepHit/fully_connected_4/kernel/Regularizer/Sum_1Sum5DeepHit/fully_connected_4/kernel/Regularizer/Square_14DeepHit/fully_connected_4/kernel/Regularizer/Const_1*
T0*

Tidx0*
	keep_dims( 
a
4DeepHit/fully_connected_4/kernel/Regularizer/mul/x_1Const*
dtype0*
valueB
 *ЈQ8
ђ
2DeepHit/fully_connected_4/kernel/Regularizer/mul_1Mul4DeepHit/fully_connected_4/kernel/Regularizer/mul/x_12DeepHit/fully_connected_4/kernel/Regularizer/Sum_1*
T0
Ь
4DeepHit/fully_connected_4/biases/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1*
dtype0*
valueB*    
б
"DeepHit/fully_connected_4/biases_1VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:*1
shared_name" DeepHit/fully_connected_4/biases
{
ADeepHit/fully_connected_4/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOp"DeepHit/fully_connected_4/biases_1
Ї
)DeepHit/fully_connected_4/biases/Assign_1AssignVariableOp"DeepHit/fully_connected_4/biases_14DeepHit/fully_connected_4/biases/Initializer/zeros_1*
dtype0*
validate_shape( 
w
4DeepHit/fully_connected_4/biases/Read/ReadVariableOpReadVariableOp"DeepHit/fully_connected_4/biases_1*
dtype0
s
/DeepHit/fully_connected_4/MatMul/ReadVariableOpReadVariableOp#DeepHit/fully_connected_4/weights_1*
dtype0
Ю
"DeepHit/fully_connected_4/MatMul_1MatMulDeepHit/concat_1/DeepHit/fully_connected_4/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b( 
s
0DeepHit/fully_connected_4/BiasAdd/ReadVariableOpReadVariableOp"DeepHit/fully_connected_4/biases_1*
dtype0
§
#DeepHit/fully_connected_4/BiasAdd_1BiasAdd"DeepHit/fully_connected_4/MatMul_10DeepHit/fully_connected_4/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
T
DeepHit/fully_connected_4/Elu_1Elu#DeepHit/fully_connected_4/BiasAdd_1*
T0
w
DeepHit/stack_3PackDeepHit/fully_connected_3/Elu_1DeepHit/fully_connected_4/Elu_1*
N*
T0*

axis
L
DeepHit/Reshape/shape_1Const*
dtype0*
valueB"€€€€(   
^
DeepHit/Reshape_10ReshapeDeepHit/stack_3DeepHit/Reshape/shape_1*
T0*
Tshape0
>
DeepHit/sub_2/x_1Const*
dtype0*
valueB
 *  А?
N
DeepHit/sub_2_1SubDeepHit/sub_2/x_1DeepHit/keep_probability_1*
T0
F
DeepHit/dropout_2/Const_1Const*
dtype0*
valueB
 *  А?
S
DeepHit/dropout_2/Sub_1SubDeepHit/dropout_2/Const_1DeepHit/sub_2_1*
T0
\
DeepHit/dropout_2/RealDiv_1RealDivDeepHit/Reshape_10DeepHit/dropout_2/Sub_1*
T0
O
DeepHit/dropout_2/Shape_1ShapeDeepHit/Reshape_10*
T0*
out_type0
Л
0DeepHit/dropout_2/random_uniform/RandomUniform_1RandomUniformDeepHit/dropout_2/Shape_1*
T0*
dtype0*

seed *
seed2 
|
 DeepHit/dropout_2/GreaterEqual_1GreaterEqual0DeepHit/dropout_2/random_uniform/RandomUniform_1DeepHit/sub_2_1*
T0
j
DeepHit/dropout_2/Cast_1Cast DeepHit/dropout_2/GreaterEqual_1*

DstT0*

SrcT0
*
Truncate( 
^
DeepHit/dropout_2/Mul_1MulDeepHit/dropout_2/RealDiv_1DeepHit/dropout_2/Cast_1*
T0
Щ
7DeepHit/Output/weights/Initializer/random_uniform/shapeConst*+
_class!
loc:@DeepHit/Output/weights_1*
dtype0*
valueB"(     
П
5DeepHit/Output/weights/Initializer/random_uniform/minConst*+
_class!
loc:@DeepHit/Output/weights_1*
dtype0*
valueB
 *ѓл
Њ
П
5DeepHit/Output/weights/Initializer/random_uniform/maxConst*+
_class!
loc:@DeepHit/Output/weights_1*
dtype0*
valueB
 *ѓл
>
е
?DeepHit/Output/weights/Initializer/random_uniform/RandomUniformRandomUniform7DeepHit/Output/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@DeepHit/Output/weights_1*
dtype0*

seed *
seed2 
а
5DeepHit/Output/weights/Initializer/random_uniform/subSub5DeepHit/Output/weights/Initializer/random_uniform/max5DeepHit/Output/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@DeepHit/Output/weights_1
к
5DeepHit/Output/weights/Initializer/random_uniform/mulMul?DeepHit/Output/weights/Initializer/random_uniform/RandomUniform5DeepHit/Output/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@DeepHit/Output/weights_1
ё
1DeepHit/Output/weights/Initializer/random_uniformAddV25DeepHit/Output/weights/Initializer/random_uniform/mul5DeepHit/Output/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@DeepHit/Output/weights_1
»
DeepHit/Output/weights_1VarHandleOp*+
_class!
loc:@DeepHit/Output/weights_1*
allowed_devices
 *
	container *
dtype0*
shape:	(Ю*'
shared_nameDeepHit/Output/weights
g
7DeepHit/Output/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpDeepHit/Output/weights_1
£
DeepHit/Output/weights/Assign_1AssignVariableOpDeepHit/Output/weights_11DeepHit/Output/weights/Initializer/random_uniform*
dtype0*
validate_shape( 
c
*DeepHit/Output/weights/Read/ReadVariableOpReadVariableOpDeepHit/Output/weights_1*
dtype0
m
4DeepHit/Output/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpDeepHit/Output/weights_1*
dtype0
m
'DeepHit/Output/kernel/Regularizer/Abs_1Abs4DeepHit/Output/kernel/Regularizer/Abs/ReadVariableOp*
T0
^
)DeepHit/Output/kernel/Regularizer/Const_1Const*
dtype0*
valueB"       
®
'DeepHit/Output/kernel/Regularizer/Sum_1Sum'DeepHit/Output/kernel/Regularizer/Abs_1)DeepHit/Output/kernel/Regularizer/Const_1*
T0*

Tidx0*
	keep_dims( 
V
)DeepHit/Output/kernel/Regularizer/mul/x_1Const*
dtype0*
valueB
 *Ј—8
Л
'DeepHit/Output/kernel/Regularizer/mul_1Mul)DeepHit/Output/kernel/Regularizer/mul/x_1'DeepHit/Output/kernel/Regularizer/Sum_1*
T0
З
)DeepHit/Output/biases/Initializer/zeros_1Const**
_class 
loc:@DeepHit/Output/biases_1*
dtype0*
valueBЮ*    
Ѕ
DeepHit/Output/biases_1VarHandleOp**
_class 
loc:@DeepHit/Output/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:Ю*&
shared_nameDeepHit/Output/biases
e
6DeepHit/Output/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpDeepHit/Output/biases_1
Щ
DeepHit/Output/biases/Assign_1AssignVariableOpDeepHit/Output/biases_1)DeepHit/Output/biases/Initializer/zeros_1*
dtype0*
validate_shape( 
a
)DeepHit/Output/biases/Read/ReadVariableOpReadVariableOpDeepHit/Output/biases_1*
dtype0
]
$DeepHit/Output/MatMul/ReadVariableOpReadVariableOpDeepHit/Output/weights_1*
dtype0
П
DeepHit/Output/MatMul_1MatMulDeepHit/dropout_2/Mul_1$DeepHit/Output/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b( 
]
%DeepHit/Output/BiasAdd/ReadVariableOpReadVariableOpDeepHit/Output/biases_1*
dtype0
Г
DeepHit/Output/BiasAdd_1BiasAddDeepHit/Output/MatMul_1%DeepHit/Output/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
F
DeepHit/Output/Softmax_1SoftmaxDeepHit/Output/BiasAdd_1*
T0
R
DeepHit/Reshape_1/shape_1Const*
dtype0*!
valueB"€€€€   П   
j
DeepHit/Reshape_1_1ReshapeDeepHit/Output/Softmax_1DeepHit/Reshape_1/shape_1*
T0*
Tshape0
1
DeepHit/Sign_3SignDeepHit/labels_1*
T0
D
DeepHit/mul_12MulDeepHit/mask1_1DeepHit/Reshape_1_1*
T0
I
DeepHit/Sum/reduction_indices_1Const*
dtype0*
value	B :
k
DeepHit/Sum_8SumDeepHit/mul_12DeepHit/Sum/reduction_indices_1*
T0*

Tidx0*
	keep_dims( 
K
!DeepHit/Sum_1/reduction_indices_1Const*
dtype0*
value	B :
n
DeepHit/Sum_1_1SumDeepHit/Sum_8!DeepHit/Sum_1/reduction_indices_1*
T0*

Tidx0*
	keep_dims(
<
DeepHit/add/y_1Const*
dtype0*
valueB
 *wћ+2
A
DeepHit/add_6AddV2DeepHit/Sum_1_1DeepHit/add/y_1*
T0
,
DeepHit/Log_2LogDeepHit/add_6*
T0
>
DeepHit/mul_1_1MulDeepHit/Sign_3DeepHit/Log_2*
T0
E
DeepHit/mul_2_1MulDeepHit/mask1_1DeepHit/Reshape_1_1*
T0
K
!DeepHit/Sum_2/reduction_indices_1Const*
dtype0*
value	B :
p
DeepHit/Sum_2_1SumDeepHit/mul_2_1!DeepHit/Sum_2/reduction_indices_1*
T0*

Tidx0*
	keep_dims( 
K
!DeepHit/Sum_3/reduction_indices_1Const*
dtype0*
value	B :
p
DeepHit/Sum_3_1SumDeepHit/Sum_2_1!DeepHit/Sum_3/reduction_indices_1*
T0*

Tidx0*
	keep_dims(
>
DeepHit/sub_3/x_1Const*
dtype0*
valueB
 *  А?
B
DeepHit/sub_3_1SubDeepHit/sub_3/x_1DeepHit/Sign_3*
T0
>
DeepHit/add_1/y_1Const*
dtype0*
valueB
 *wћ+2
E
DeepHit/add_1_1AddV2DeepHit/Sum_3_1DeepHit/add_1/y_1*
T0
0
DeepHit/Log_1_1LogDeepHit/add_1_1*
T0
A
DeepHit/mul_3_1MulDeepHit/sub_3_1DeepHit/Log_1_1*
T0
>
DeepHit/mul_4/x_1Const*
dtype0*
valueB
 *  А?
C
DeepHit/mul_4_1MulDeepHit/mul_4/x_1DeepHit/mul_3_1*
T0
C
DeepHit/add_2_1AddV2DeepHit/mul_1_1DeepHit/mul_4_1*
T0
D
DeepHit/Const_4Const*
dtype0*
valueB"       
^
DeepHit/Mean_7MeanDeepHit/add_2_1DeepHit/Const_4*
T0*

Tidx0*
	keep_dims( 
-
DeepHit/Neg_3NegDeepHit/Mean_7*
T0
>
DeepHit/Const_1_1Const*
dtype0*
valueB
 *Ќћћ=
S
DeepHit/ones_like/Shape_1ShapeDeepHit/timetoevents_1*
T0*
out_type0
F
DeepHit/ones_like/Const_1Const*
dtype0*
valueB
 *  А?
l
DeepHit/ones_like_4FillDeepHit/ones_like/Shape_1DeepHit/ones_like/Const_1*
T0*

index_type0
>
DeepHit/Equal/y_1Const*
dtype0*
valueB
 *  А?
f
DeepHit/Equal_4EqualDeepHit/labels_1DeepHit/Equal/y_1*
T0*
incompatible_shape_error(
O
DeepHit/Cast_4CastDeepHit/Equal_4*

DstT0*

SrcT0
*
Truncate( 
I
DeepHit/Squeeze_2SqueezeDeepHit/Cast_4*
T0*
squeeze_dims
 
2
DeepHit/Diag_2DiagDeepHit/Squeeze_2*
T0
N
DeepHit/Slice/begin_1Const*
dtype0*!
valueB"            
M
DeepHit/Slice/size_1Const*
dtype0*!
valueB"€€€€   €€€€
p
DeepHit/Slice_4SliceDeepHit/Reshape_1_1DeepHit/Slice/begin_1DeepHit/Slice/size_1*
Index0*
T0
N
DeepHit/Reshape_2/shape_1Const*
dtype0*
valueB"€€€€П   
a
DeepHit/Reshape_2_1ReshapeDeepHit/Slice_4DeepHit/Reshape_2/shape_1*
T0*
Tshape0
M
DeepHit/transpose/perm_1Const*
dtype0*
valueB"       
b
DeepHit/transpose_10	TransposeDeepHit/mask2_1DeepHit/transpose/perm_1*
T0*
Tperm0
u
DeepHit/MatMul_10MatMulDeepHit/Reshape_2_1DeepHit/transpose_10*
T0*
transpose_a( *
transpose_b( 
:
DeepHit/DiagPart_2DiagPartDeepHit/MatMul_10*
T0
N
DeepHit/Reshape_3/shape_1Const*
dtype0*
valueB"€€€€   
d
DeepHit/Reshape_3_1ReshapeDeepHit/DiagPart_2DeepHit/Reshape_3/shape_1*
T0*
Tshape0
O
DeepHit/transpose_1/perm_1Const*
dtype0*
valueB"       
i
DeepHit/transpose_1_1	TransposeDeepHit/Reshape_3_1DeepHit/transpose_1/perm_1*
T0*
Tperm0
w
DeepHit/MatMul_1_1MatMulDeepHit/ones_like_4DeepHit/transpose_1_1*
T0*
transpose_a( *
transpose_b( 
F
DeepHit/sub_4_1SubDeepHit/MatMul_1_1DeepHit/MatMul_10*
T0
O
DeepHit/transpose_2/perm_1Const*
dtype0*
valueB"       
e
DeepHit/transpose_2_1	TransposeDeepHit/sub_4_1DeepHit/transpose_2/perm_1*
T0*
Tperm0
O
DeepHit/transpose_3/perm_1Const*
dtype0*
valueB"       
l
DeepHit/transpose_3_1	TransposeDeepHit/timetoevents_1DeepHit/transpose_3/perm_1*
T0*
Tperm0
w
DeepHit/MatMul_2_1MatMulDeepHit/ones_like_4DeepHit/transpose_3_1*
T0*
transpose_a( *
transpose_b( 
O
DeepHit/transpose_4/perm_1Const*
dtype0*
valueB"       
i
DeepHit/transpose_4_1	TransposeDeepHit/ones_like_4DeepHit/transpose_4/perm_1*
T0*
Tperm0
z
DeepHit/MatMul_3_1MatMulDeepHit/timetoevents_1DeepHit/transpose_4_1*
T0*
transpose_a( *
transpose_b( 
G
DeepHit/sub_5_1SubDeepHit/MatMul_2_1DeepHit/MatMul_3_1*
T0
2
DeepHit/Sign_1_1SignDeepHit/sub_5_1*
T0
1
DeepHit/Relu_2ReluDeepHit/Sign_1_1*
T0
f
DeepHit/MatMul_4_1BatchMatMulV2DeepHit/Diag_2DeepHit/Relu_2*
T0*
adj_x( *
adj_y( 
6
DeepHit/Neg_1_1NegDeepHit/transpose_2_1*
T0
I
DeepHit/truediv_2RealDivDeepHit/Neg_1_1DeepHit/Const_1_1*
T0
0
DeepHit/Exp_2ExpDeepHit/truediv_2*
T0
B
DeepHit/mul_5_1MulDeepHit/MatMul_4_1DeepHit/Exp_2*
T0
L
"DeepHit/Mean_1/reduction_indices_1Const*
dtype0*
value	B :
s
DeepHit/Mean_1_1MeanDeepHit/mul_5_1"DeepHit/Mean_1/reduction_indices_1*
T0*

Tidx0*
	keep_dims(
U
DeepHit/ones_like_1/Shape_1ShapeDeepHit/timetoevents_1*
T0*
out_type0
H
DeepHit/ones_like_1/Const_1Const*
dtype0*
valueB
 *  А?
r
DeepHit/ones_like_1_1FillDeepHit/ones_like_1/Shape_1DeepHit/ones_like_1/Const_1*
T0*

index_type0
@
DeepHit/Equal_1/y_1Const*
dtype0*
valueB
 *   @
j
DeepHit/Equal_1_1EqualDeepHit/labels_1DeepHit/Equal_1/y_1*
T0*
incompatible_shape_error(
S
DeepHit/Cast_1_1CastDeepHit/Equal_1_1*

DstT0*

SrcT0
*
Truncate( 
M
DeepHit/Squeeze_1_1SqueezeDeepHit/Cast_1_1*
T0*
squeeze_dims
 
6
DeepHit/Diag_1_1DiagDeepHit/Squeeze_1_1*
T0
P
DeepHit/Slice_1/begin_1Const*
dtype0*!
valueB"           
O
DeepHit/Slice_1/size_1Const*
dtype0*!
valueB"€€€€   €€€€
v
DeepHit/Slice_1_1SliceDeepHit/Reshape_1_1DeepHit/Slice_1/begin_1DeepHit/Slice_1/size_1*
Index0*
T0
N
DeepHit/Reshape_4/shape_1Const*
dtype0*
valueB"€€€€П   
c
DeepHit/Reshape_4_1ReshapeDeepHit/Slice_1_1DeepHit/Reshape_4/shape_1*
T0*
Tshape0
O
DeepHit/transpose_5/perm_1Const*
dtype0*
valueB"       
e
DeepHit/transpose_5_1	TransposeDeepHit/mask2_1DeepHit/transpose_5/perm_1*
T0*
Tperm0
w
DeepHit/MatMul_5_1MatMulDeepHit/Reshape_4_1DeepHit/transpose_5_1*
T0*
transpose_a( *
transpose_b( 
=
DeepHit/DiagPart_1_1DiagPartDeepHit/MatMul_5_1*
T0
N
DeepHit/Reshape_5/shape_1Const*
dtype0*
valueB"€€€€   
f
DeepHit/Reshape_5_1ReshapeDeepHit/DiagPart_1_1DeepHit/Reshape_5/shape_1*
T0*
Tshape0
O
DeepHit/transpose_6/perm_1Const*
dtype0*
valueB"       
i
DeepHit/transpose_6_1	TransposeDeepHit/Reshape_5_1DeepHit/transpose_6/perm_1*
T0*
Tperm0
y
DeepHit/MatMul_6_1MatMulDeepHit/ones_like_1_1DeepHit/transpose_6_1*
T0*
transpose_a( *
transpose_b( 
G
DeepHit/sub_6_1SubDeepHit/MatMul_6_1DeepHit/MatMul_5_1*
T0
O
DeepHit/transpose_7/perm_1Const*
dtype0*
valueB"       
e
DeepHit/transpose_7_1	TransposeDeepHit/sub_6_1DeepHit/transpose_7/perm_1*
T0*
Tperm0
O
DeepHit/transpose_8/perm_1Const*
dtype0*
valueB"       
l
DeepHit/transpose_8_1	TransposeDeepHit/timetoevents_1DeepHit/transpose_8/perm_1*
T0*
Tperm0
y
DeepHit/MatMul_7_1MatMulDeepHit/ones_like_1_1DeepHit/transpose_8_1*
T0*
transpose_a( *
transpose_b( 
O
DeepHit/transpose_9/perm_1Const*
dtype0*
valueB"       
k
DeepHit/transpose_9_1	TransposeDeepHit/ones_like_1_1DeepHit/transpose_9/perm_1*
T0*
Tperm0
z
DeepHit/MatMul_8_1MatMulDeepHit/timetoevents_1DeepHit/transpose_9_1*
T0*
transpose_a( *
transpose_b( 
G
DeepHit/sub_7_1SubDeepHit/MatMul_7_1DeepHit/MatMul_8_1*
T0
2
DeepHit/Sign_2_1SignDeepHit/sub_7_1*
T0
3
DeepHit/Relu_1_1ReluDeepHit/Sign_2_1*
T0
j
DeepHit/MatMul_9_1BatchMatMulV2DeepHit/Diag_1_1DeepHit/Relu_1_1*
T0*
adj_x( *
adj_y( 
6
DeepHit/Neg_2_1NegDeepHit/transpose_7_1*
T0
K
DeepHit/truediv_1_1RealDivDeepHit/Neg_2_1DeepHit/Const_1_1*
T0
4
DeepHit/Exp_1_1ExpDeepHit/truediv_1_1*
T0
D
DeepHit/mul_6_1MulDeepHit/MatMul_9_1DeepHit/Exp_1_1*
T0
L
"DeepHit/Mean_2/reduction_indices_1Const*
dtype0*
value	B :
s
DeepHit/Mean_2_1MeanDeepHit/mul_6_1"DeepHit/Mean_2/reduction_indices_1*
T0*

Tidx0*
	keep_dims(
[
DeepHit/stack_1_1PackDeepHit/Mean_1_1DeepHit/Mean_2_1*
N*
T0*

axis
N
DeepHit/Reshape_6/shape_1Const*
dtype0*
valueB"€€€€   
c
DeepHit/Reshape_6_1ReshapeDeepHit/stack_1_1DeepHit/Reshape_6/shape_1*
T0*
Tshape0
L
"DeepHit/Mean_3/reduction_indices_1Const*
dtype0*
value	B :
w
DeepHit/Mean_3_1MeanDeepHit/Reshape_6_1"DeepHit/Mean_3/reduction_indices_1*
T0*

Tidx0*
	keep_dims(
F
DeepHit/Const_2_1Const*
dtype0*
valueB"       
a
DeepHit/Sum_4_1SumDeepHit/Mean_3_1DeepHit/Const_2_1*
T0*

Tidx0*
	keep_dims( 
U
DeepHit/ones_like_2/Shape_1ShapeDeepHit/timetoevents_1*
T0*
out_type0
H
DeepHit/ones_like_2/Const_1Const*
dtype0*
valueB
 *  А?
r
DeepHit/ones_like_2_1FillDeepHit/ones_like_2/Shape_1DeepHit/ones_like_2/Const_1*
T0*

index_type0
@
DeepHit/Equal_2/y_1Const*
dtype0*
valueB
 *  А?
j
DeepHit/Equal_2_1EqualDeepHit/labels_1DeepHit/Equal_2/y_1*
T0*
incompatible_shape_error(
S
DeepHit/Cast_2_1CastDeepHit/Equal_2_1*

DstT0*

SrcT0
*
Truncate( 
P
DeepHit/Slice_2/begin_1Const*
dtype0*!
valueB"            
O
DeepHit/Slice_2/size_1Const*
dtype0*!
valueB"€€€€   €€€€
v
DeepHit/Slice_2_1SliceDeepHit/Reshape_1_1DeepHit/Slice_2/begin_1DeepHit/Slice_2/size_1*
Index0*
T0
N
DeepHit/Reshape_7/shape_1Const*
dtype0*
valueB"€€€€П   
c
DeepHit/Reshape_7_1ReshapeDeepHit/Slice_2_1DeepHit/Reshape_7/shape_1*
T0*
Tshape0
E
DeepHit/mul_7_1MulDeepHit/Reshape_7_1DeepHit/mask2_1*
T0
K
!DeepHit/Sum_5/reduction_indices_1Const*
dtype0*
value	B : 
p
DeepHit/Sum_5_1SumDeepHit/mul_7_1!DeepHit/Sum_5/reduction_indices_1*
T0*

Tidx0*
	keep_dims( 
B
DeepHit/sub_8_1SubDeepHit/Sum_5_1DeepHit/Cast_2_1*
T0
<
DeepHit/pow/y_1Const*
dtype0*
valueB
 *   @
?
DeepHit/pow_2PowDeepHit/sub_8_1DeepHit/pow/y_1*
T0
L
"DeepHit/Mean_4/reduction_indices_1Const*
dtype0*
value	B :
q
DeepHit/Mean_4_1MeanDeepHit/pow_2"DeepHit/Mean_4/reduction_indices_1*
T0*

Tidx0*
	keep_dims(
U
DeepHit/ones_like_3/Shape_1ShapeDeepHit/timetoevents_1*
T0*
out_type0
H
DeepHit/ones_like_3/Const_1Const*
dtype0*
valueB
 *  А?
r
DeepHit/ones_like_3_1FillDeepHit/ones_like_3/Shape_1DeepHit/ones_like_3/Const_1*
T0*

index_type0
@
DeepHit/Equal_3/y_1Const*
dtype0*
valueB
 *   @
j
DeepHit/Equal_3_1EqualDeepHit/labels_1DeepHit/Equal_3/y_1*
T0*
incompatible_shape_error(
S
DeepHit/Cast_3_1CastDeepHit/Equal_3_1*

DstT0*

SrcT0
*
Truncate( 
P
DeepHit/Slice_3/begin_1Const*
dtype0*!
valueB"           
O
DeepHit/Slice_3/size_1Const*
dtype0*!
valueB"€€€€   €€€€
v
DeepHit/Slice_3_1SliceDeepHit/Reshape_1_1DeepHit/Slice_3/begin_1DeepHit/Slice_3/size_1*
Index0*
T0
N
DeepHit/Reshape_8/shape_1Const*
dtype0*
valueB"€€€€П   
c
DeepHit/Reshape_8_1ReshapeDeepHit/Slice_3_1DeepHit/Reshape_8/shape_1*
T0*
Tshape0
E
DeepHit/mul_8_1MulDeepHit/Reshape_8_1DeepHit/mask2_1*
T0
K
!DeepHit/Sum_6/reduction_indices_1Const*
dtype0*
value	B : 
p
DeepHit/Sum_6_1SumDeepHit/mul_8_1!DeepHit/Sum_6/reduction_indices_1*
T0*

Tidx0*
	keep_dims( 
B
DeepHit/sub_9_1SubDeepHit/Sum_6_1DeepHit/Cast_3_1*
T0
>
DeepHit/pow_1/y_1Const*
dtype0*
valueB
 *   @
C
DeepHit/pow_1_1PowDeepHit/sub_9_1DeepHit/pow_1/y_1*
T0
L
"DeepHit/Mean_5/reduction_indices_1Const*
dtype0*
value	B :
s
DeepHit/Mean_5_1MeanDeepHit/pow_1_1"DeepHit/Mean_5/reduction_indices_1*
T0*

Tidx0*
	keep_dims(
[
DeepHit/stack_2_1PackDeepHit/Mean_4_1DeepHit/Mean_5_1*
N*
T0*

axis
N
DeepHit/Reshape_9/shape_1Const*
dtype0*
valueB"€€€€   
c
DeepHit/Reshape_9_1ReshapeDeepHit/stack_2_1DeepHit/Reshape_9/shape_1*
T0*
Tshape0
L
"DeepHit/Mean_6/reduction_indices_1Const*
dtype0*
value	B :
w
DeepHit/Mean_6_1MeanDeepHit/Reshape_9_1"DeepHit/Mean_6/reduction_indices_1*
T0*

Tidx0*
	keep_dims(
F
DeepHit/Const_3_1Const*
dtype0*
valueB"       
a
DeepHit/Sum_7_1SumDeepHit/Mean_6_1DeepHit/Const_3_1*
T0*

Tidx0*
	keep_dims( 
?
DeepHit/mul_9_1MulDeepHit/alpha_1DeepHit/Neg_3*
T0
A
DeepHit/mul_10_1MulDeepHit/beta_1DeepHit/Sum_4_1*
T0
D
DeepHit/add_3_1AddV2DeepHit/mul_9_1DeepHit/mul_10_1*
T0
B
DeepHit/mul_11_1MulDeepHit/gamma_1DeepHit/Sum_7_1*
T0
D
DeepHit/add_4_1AddV2DeepHit/add_3_1DeepHit/mul_11_1*
T0
и
#DeepHit/total_regularization_loss_1AddN0DeepHit/fully_connected/kernel/Regularizer/mul_12DeepHit/fully_connected_1/kernel/Regularizer/mul_12DeepHit/fully_connected_2/kernel/Regularizer/mul_12DeepHit/fully_connected_3/kernel/Regularizer/mul_12DeepHit/fully_connected_4/kernel/Regularizer/mul_1'DeepHit/Output/kernel/Regularizer/mul_1*
N*
T0
W
DeepHit/add_5_1AddV2DeepHit/add_4_1#DeepHit/total_regularization_loss_1*
T0
B
DeepHit/gradients/Shape_1Const*
dtype0*
valueB 
P
#DeepHit/gradients/grad_ys_0/Const_1Const*
dtype0*
valueB
 *  А?
А
DeepHit/gradients/grad_ys_0_1FillDeepHit/gradients/Shape_1#DeepHit/gradients/grad_ys_0/Const_1*
T0*

index_type0
_
7DeepHit/gradients/DeepHit/add_5_grad/tuple/group_deps_1NoOp^DeepHit/gradients/grad_ys_0_1
я
?DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_2IdentityDeepHit/gradients/grad_ys_0_18^DeepHit/gradients/DeepHit/add_5_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
б
ADeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1_1IdentityDeepHit/gradients/grad_ys_0_18^DeepHit/gradients/DeepHit/add_5_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
Б
7DeepHit/gradients/DeepHit/add_4_grad/tuple/group_deps_1NoOp@^DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_2
Б
?DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency_2Identity?DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_28^DeepHit/gradients/DeepHit/add_4_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
Г
ADeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency_1_1Identity?DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_28^DeepHit/gradients/DeepHit/add_4_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
Ч
KDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps_1NoOpB^DeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1_1
Ђ
SDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_6IdentityADeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1_1L^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
≠
UDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_1_1IdentityADeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1_1L^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
≠
UDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_2_1IdentityADeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1_1L^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
≠
UDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_3_1IdentityADeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1_1L^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
≠
UDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_4_1IdentityADeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1_1L^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
≠
UDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_5_1IdentityADeepHit/gradients/DeepHit/add_5_grad/tuple/control_dependency_1_1L^DeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
Б
7DeepHit/gradients/DeepHit/add_3_grad/tuple/group_deps_1NoOp@^DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency_2
Б
?DeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependency_2Identity?DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency_28^DeepHit/gradients/DeepHit/add_3_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
Г
ADeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependency_1_1Identity?DeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency_28^DeepHit/gradients/DeepHit/add_3_grad/tuple/group_deps_1*
T0*0
_class&
$"loc:@DeepHit/gradients/grad_ys_0_1
П
+DeepHit/gradients/DeepHit/mul_11_grad/Mul_2MulADeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency_1_1DeepHit/Sum_7_1*
T0
С
-DeepHit/gradients/DeepHit/mul_11_grad/Mul_1_1MulADeepHit/gradients/DeepHit/add_4_grad/tuple/control_dependency_1_1DeepHit/gamma_1*
T0
Ю
8DeepHit/gradients/DeepHit/mul_11_grad/tuple/group_deps_1NoOp.^DeepHit/gradients/DeepHit/mul_11_grad/Mul_1_1,^DeepHit/gradients/DeepHit/mul_11_grad/Mul_2
э
@DeepHit/gradients/DeepHit/mul_11_grad/tuple/control_dependency_2Identity+DeepHit/gradients/DeepHit/mul_11_grad/Mul_29^DeepHit/gradients/DeepHit/mul_11_grad/tuple/group_deps_1*
T0*>
_class4
20loc:@DeepHit/gradients/DeepHit/mul_11_grad/Mul_2
Г
BDeepHit/gradients/DeepHit/mul_11_grad/tuple/control_dependency_1_1Identity-DeepHit/gradients/DeepHit/mul_11_grad/Mul_1_19^DeepHit/gradients/DeepHit/mul_11_grad/tuple/group_deps_1*
T0*@
_class6
42loc:@DeepHit/gradients/DeepHit/mul_11_grad/Mul_1_1
в
KDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_2MulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_60DeepHit/fully_connected/kernel/Regularizer/Sum_1*
T0
ж
MDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_1_1MulSDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_62DeepHit/fully_connected/kernel/Regularizer/mul/x_1*
T0
ю
XDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/group_deps_1NoOpN^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_1_1L^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_2
э
`DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/control_dependency_2IdentityKDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_2Y^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*^
_classT
RPloc:@DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_2
Г
bDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1IdentityMDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_1_1Y^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*`
_classV
TRloc:@DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/Mul_1_1
и
MDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_2MulUDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_1_12DeepHit/fully_connected_1/kernel/Regularizer/Sum_1*
T0
м
ODeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_1_1MulUDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_1_14DeepHit/fully_connected_1/kernel/Regularizer/mul/x_1*
T0
Д
ZDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/group_deps_1NoOpP^DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_1_1N^DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_2
Е
bDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/control_dependency_2IdentityMDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_2[^DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*`
_classV
TRloc:@DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_2
Л
dDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1IdentityODeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_1_1[^DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*b
_classX
VTloc:@DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/Mul_1_1
и
MDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_2MulUDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_2_12DeepHit/fully_connected_2/kernel/Regularizer/Sum_1*
T0
м
ODeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_1_1MulUDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_2_14DeepHit/fully_connected_2/kernel/Regularizer/mul/x_1*
T0
Д
ZDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/group_deps_1NoOpP^DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_1_1N^DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_2
Е
bDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/control_dependency_2IdentityMDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_2[^DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*`
_classV
TRloc:@DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_2
Л
dDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1IdentityODeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_1_1[^DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*b
_classX
VTloc:@DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/Mul_1_1
и
MDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_2MulUDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_3_12DeepHit/fully_connected_3/kernel/Regularizer/Sum_1*
T0
м
ODeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_1_1MulUDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_3_14DeepHit/fully_connected_3/kernel/Regularizer/mul/x_1*
T0
Д
ZDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/group_deps_1NoOpP^DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_1_1N^DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_2
Е
bDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/control_dependency_2IdentityMDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_2[^DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*`
_classV
TRloc:@DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_2
Л
dDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1IdentityODeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_1_1[^DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*b
_classX
VTloc:@DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/Mul_1_1
и
MDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_2MulUDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_4_12DeepHit/fully_connected_4/kernel/Regularizer/Sum_1*
T0
м
ODeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_1_1MulUDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_4_14DeepHit/fully_connected_4/kernel/Regularizer/mul/x_1*
T0
Д
ZDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/group_deps_1NoOpP^DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_1_1N^DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_2
Е
bDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/control_dependency_2IdentityMDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_2[^DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*`
_classV
TRloc:@DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_2
Л
dDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1IdentityODeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_1_1[^DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*b
_classX
VTloc:@DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/Mul_1_1
“
BDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_2MulUDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_5_1'DeepHit/Output/kernel/Regularizer/Sum_1*
T0
÷
DDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_1_1MulUDeepHit/gradients/DeepHit/total_regularization_loss_grad/tuple/control_dependency_5_1)DeepHit/Output/kernel/Regularizer/mul/x_1*
T0
г
ODeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/group_deps_1NoOpE^DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_1_1C^DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_2
ў
WDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/control_dependency_2IdentityBDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_2P^DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*U
_classK
IGloc:@DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_2
я
YDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1IdentityDDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_1_1P^DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/group_deps_1*
T0*W
_classM
KIloc:@DeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/Mul_1_1
К
*DeepHit/gradients/DeepHit/mul_9_grad/Mul_2Mul?DeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependency_2DeepHit/Neg_3*
T0
О
,DeepHit/gradients/DeepHit/mul_9_grad/Mul_1_1Mul?DeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependency_2DeepHit/alpha_1*
T0
Ы
7DeepHit/gradients/DeepHit/mul_9_grad/tuple/group_deps_1NoOp-^DeepHit/gradients/DeepHit/mul_9_grad/Mul_1_1+^DeepHit/gradients/DeepHit/mul_9_grad/Mul_2
щ
?DeepHit/gradients/DeepHit/mul_9_grad/tuple/control_dependency_2Identity*DeepHit/gradients/DeepHit/mul_9_grad/Mul_28^DeepHit/gradients/DeepHit/mul_9_grad/tuple/group_deps_1*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/mul_9_grad/Mul_2
€
ADeepHit/gradients/DeepHit/mul_9_grad/tuple/control_dependency_1_1Identity,DeepHit/gradients/DeepHit/mul_9_grad/Mul_1_18^DeepHit/gradients/DeepHit/mul_9_grad/tuple/group_deps_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_9_grad/Mul_1_1
П
+DeepHit/gradients/DeepHit/mul_10_grad/Mul_2MulADeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependency_1_1DeepHit/Sum_4_1*
T0
Р
-DeepHit/gradients/DeepHit/mul_10_grad/Mul_1_1MulADeepHit/gradients/DeepHit/add_3_grad/tuple/control_dependency_1_1DeepHit/beta_1*
T0
Ю
8DeepHit/gradients/DeepHit/mul_10_grad/tuple/group_deps_1NoOp.^DeepHit/gradients/DeepHit/mul_10_grad/Mul_1_1,^DeepHit/gradients/DeepHit/mul_10_grad/Mul_2
э
@DeepHit/gradients/DeepHit/mul_10_grad/tuple/control_dependency_2Identity+DeepHit/gradients/DeepHit/mul_10_grad/Mul_29^DeepHit/gradients/DeepHit/mul_10_grad/tuple/group_deps_1*
T0*>
_class4
20loc:@DeepHit/gradients/DeepHit/mul_10_grad/Mul_2
Г
BDeepHit/gradients/DeepHit/mul_10_grad/tuple/control_dependency_1_1Identity-DeepHit/gradients/DeepHit/mul_10_grad/Mul_1_19^DeepHit/gradients/DeepHit/mul_10_grad/tuple/group_deps_1*
T0*@
_class6
42loc:@DeepHit/gradients/DeepHit/mul_10_grad/Mul_1_1
i
4DeepHit/gradients/DeepHit/Sum_7_grad/Reshape/shape_1Const*
dtype0*
valueB"      
 
.DeepHit/gradients/DeepHit/Sum_7_grad/Reshape_1ReshapeBDeepHit/gradients/DeepHit/mul_11_grad/tuple/control_dependency_1_14DeepHit/gradients/DeepHit/Sum_7_grad/Reshape/shape_1*
T0*
Tshape0
`
,DeepHit/gradients/DeepHit/Sum_7_grad/Shape_1ShapeDeepHit/Mean_6_1*
T0*
out_type0
ђ
+DeepHit/gradients/DeepHit/Sum_7_grad/Tile_1Tile.DeepHit/gradients/DeepHit/Sum_7_grad/Reshape_1,DeepHit/gradients/DeepHit/Sum_7_grad/Shape_1*
T0*

Tmultiples0
К
UDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Reshape/shape_1Const*
dtype0*
valueB"      
ђ
ODeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Reshape_1ReshapebDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1UDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Reshape/shape_1*
T0*
Tshape0
В
MDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Const_1Const*
dtype0*
valueB"`   
   
П
LDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Tile_1TileODeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Reshape_1MDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Const_1*
T0*

Tmultiples0
М
WDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Reshape/shape_1Const*
dtype0*
valueB"      
≤
QDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Reshape_1ReshapedDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1WDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Reshape/shape_1*
T0*
Tshape0
Д
ODeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Const_1Const*
dtype0*
valueB"
   
   
Х
NDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Tile_1TileQDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Reshape_1ODeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Const_1*
T0*

Tmultiples0
М
WDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Reshape/shape_1Const*
dtype0*
valueB"      
≤
QDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Reshape_1ReshapedDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1WDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Reshape/shape_1*
T0*
Tshape0
Д
ODeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Const_1Const*
dtype0*
valueB"
   
   
Х
NDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Tile_1TileQDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Reshape_1ODeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Const_1*
T0*

Tmultiples0
М
WDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Reshape/shape_1Const*
dtype0*
valueB"      
≤
QDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Reshape_1ReshapedDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1WDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Reshape/shape_1*
T0*
Tshape0
Д
ODeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Const_1Const*
dtype0*
valueB"j      
Х
NDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Tile_1TileQDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Reshape_1ODeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Const_1*
T0*

Tmultiples0
М
WDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Reshape/shape_1Const*
dtype0*
valueB"      
≤
QDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Reshape_1ReshapedDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1WDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Reshape/shape_1*
T0*
Tshape0
Д
ODeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Const_1Const*
dtype0*
valueB"j      
Х
NDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Tile_1TileQDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Reshape_1ODeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Const_1*
T0*

Tmultiples0
Б
LDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Reshape/shape_1Const*
dtype0*
valueB"      
С
FDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Reshape_1ReshapeYDeepHit/gradients/DeepHit/Output/kernel/Regularizer/mul_grad/tuple/control_dependency_1_1LDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Reshape/shape_1*
T0*
Tshape0
y
DDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Const_1Const*
dtype0*
valueB"(     
ф
CDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Tile_1TileFDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Reshape_1DDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Const_1*
T0*

Tmultiples0
{
(DeepHit/gradients/DeepHit/Neg_grad/Neg_1NegADeepHit/gradients/DeepHit/mul_9_grad/tuple/control_dependency_1_1*
T0
i
4DeepHit/gradients/DeepHit/Sum_4_grad/Reshape/shape_1Const*
dtype0*
valueB"      
 
.DeepHit/gradients/DeepHit/Sum_4_grad/Reshape_1ReshapeBDeepHit/gradients/DeepHit/mul_10_grad/tuple/control_dependency_1_14DeepHit/gradients/DeepHit/Sum_4_grad/Reshape/shape_1*
T0*
Tshape0
`
,DeepHit/gradients/DeepHit/Sum_4_grad/Shape_1ShapeDeepHit/Mean_3_1*
T0*
out_type0
ђ
+DeepHit/gradients/DeepHit/Sum_4_grad/Tile_1Tile.DeepHit/gradients/DeepHit/Sum_4_grad/Reshape_1,DeepHit/gradients/DeepHit/Sum_4_grad/Shape_1*
T0*

Tmultiples0
d
-DeepHit/gradients/DeepHit/Mean_6_grad/Shape_3ShapeDeepHit/Reshape_9_1*
T0*
out_type0
≥
3DeepHit/gradients/DeepHit/Mean_6_grad/BroadcastTo_1BroadcastTo+DeepHit/gradients/DeepHit/Sum_7_grad/Tile_1-DeepHit/gradients/DeepHit/Mean_6_grad/Shape_3*
T0*

Tidx0
f
/DeepHit/gradients/DeepHit/Mean_6_grad/Shape_1_1ShapeDeepHit/Reshape_9_1*
T0*
out_type0
c
/DeepHit/gradients/DeepHit/Mean_6_grad/Shape_2_1ShapeDeepHit/Mean_6_1*
T0*
out_type0
[
-DeepHit/gradients/DeepHit/Mean_6_grad/Const_2Const*
dtype0*
valueB: 
Ї
,DeepHit/gradients/DeepHit/Mean_6_grad/Prod_2Prod/DeepHit/gradients/DeepHit/Mean_6_grad/Shape_1_1-DeepHit/gradients/DeepHit/Mean_6_grad/Const_2*
T0*

Tidx0*
	keep_dims( 
]
/DeepHit/gradients/DeepHit/Mean_6_grad/Const_1_1Const*
dtype0*
valueB: 
Њ
.DeepHit/gradients/DeepHit/Mean_6_grad/Prod_1_1Prod/DeepHit/gradients/DeepHit/Mean_6_grad/Shape_2_1/DeepHit/gradients/DeepHit/Mean_6_grad/Const_1_1*
T0*

Tidx0*
	keep_dims( 
[
1DeepHit/gradients/DeepHit/Mean_6_grad/Maximum/y_1Const*
dtype0*
value	B :
¶
/DeepHit/gradients/DeepHit/Mean_6_grad/Maximum_1Maximum.DeepHit/gradients/DeepHit/Mean_6_grad/Prod_1_11DeepHit/gradients/DeepHit/Mean_6_grad/Maximum/y_1*
T0
§
0DeepHit/gradients/DeepHit/Mean_6_grad/floordiv_1FloorDiv,DeepHit/gradients/DeepHit/Mean_6_grad/Prod_2/DeepHit/gradients/DeepHit/Mean_6_grad/Maximum_1*
T0
О
,DeepHit/gradients/DeepHit/Mean_6_grad/Cast_1Cast0DeepHit/gradients/DeepHit/Mean_6_grad/floordiv_1*

DstT0*

SrcT0*
Truncate( 
¶
/DeepHit/gradients/DeepHit/Mean_6_grad/truediv_1RealDiv3DeepHit/gradients/DeepHit/Mean_6_grad/BroadcastTo_1,DeepHit/gradients/DeepHit/Mean_6_grad/Cast_1*
T0
ћ
PDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Const_1ConstM^DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Tile_1*
dtype0*
valueB
 *   @
т
NDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Mul_2Mul@DeepHit/fully_connected/kernel/Regularizer/Square/ReadVariableOpPDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Const_1*
T0
ю
PDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Mul_1_1MulLDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Sum_grad/Tile_1NDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Mul_2*
T0
–
RDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Const_1ConstO^DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Tile_1*
dtype0*
valueB
 *   @
ш
PDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Mul_2MulBDeepHit/fully_connected_1/kernel/Regularizer/Square/ReadVariableOpRDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Const_1*
T0
Д
RDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Mul_1_1MulNDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Sum_grad/Tile_1PDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Mul_2*
T0
–
RDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Const_1ConstO^DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Tile_1*
dtype0*
valueB
 *   @
ш
PDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Mul_2MulBDeepHit/fully_connected_2/kernel/Regularizer/Square/ReadVariableOpRDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Const_1*
T0
Д
RDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Mul_1_1MulNDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Sum_grad/Tile_1PDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Mul_2*
T0
–
RDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Const_1ConstO^DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Tile_1*
dtype0*
valueB
 *   @
ш
PDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Mul_2MulBDeepHit/fully_connected_3/kernel/Regularizer/Square/ReadVariableOpRDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Const_1*
T0
Д
RDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Mul_1_1MulNDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Sum_grad/Tile_1PDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Mul_2*
T0
–
RDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Const_1ConstO^DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Tile_1*
dtype0*
valueB
 *   @
ш
PDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Mul_2MulBDeepHit/fully_connected_4/kernel/Regularizer/Square/ReadVariableOpRDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Const_1*
T0
Д
RDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Mul_1_1MulNDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Sum_grad/Tile_1PDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Mul_2*
T0
К
CDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Abs_grad/Sign_1Sign4DeepHit/Output/kernel/Regularizer/Abs/ReadVariableOp*
T0
№
BDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Abs_grad/mul_1MulCDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Sum_grad/Tile_1CDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Abs_grad/Sign_1*
T0
h
3DeepHit/gradients/DeepHit/Mean_grad/Reshape/shape_1Const*
dtype0*
valueB"      
Ѓ
-DeepHit/gradients/DeepHit/Mean_grad/Reshape_1Reshape(DeepHit/gradients/DeepHit/Neg_grad/Neg_13DeepHit/gradients/DeepHit/Mean_grad/Reshape/shape_1*
T0*
Tshape0
^
+DeepHit/gradients/DeepHit/Mean_grad/Shape_3ShapeDeepHit/add_2_1*
T0*
out_type0
©
*DeepHit/gradients/DeepHit/Mean_grad/Tile_1Tile-DeepHit/gradients/DeepHit/Mean_grad/Reshape_1+DeepHit/gradients/DeepHit/Mean_grad/Shape_3*
T0*

Tmultiples0
`
-DeepHit/gradients/DeepHit/Mean_grad/Shape_1_1ShapeDeepHit/add_2_1*
T0*
out_type0
V
-DeepHit/gradients/DeepHit/Mean_grad/Shape_2_1Const*
dtype0*
valueB 
Y
+DeepHit/gradients/DeepHit/Mean_grad/Const_2Const*
dtype0*
valueB: 
і
*DeepHit/gradients/DeepHit/Mean_grad/Prod_2Prod-DeepHit/gradients/DeepHit/Mean_grad/Shape_1_1+DeepHit/gradients/DeepHit/Mean_grad/Const_2*
T0*

Tidx0*
	keep_dims( 
[
-DeepHit/gradients/DeepHit/Mean_grad/Const_1_1Const*
dtype0*
valueB: 
Є
,DeepHit/gradients/DeepHit/Mean_grad/Prod_1_1Prod-DeepHit/gradients/DeepHit/Mean_grad/Shape_2_1-DeepHit/gradients/DeepHit/Mean_grad/Const_1_1*
T0*

Tidx0*
	keep_dims( 
Y
/DeepHit/gradients/DeepHit/Mean_grad/Maximum/y_1Const*
dtype0*
value	B :
†
-DeepHit/gradients/DeepHit/Mean_grad/Maximum_1Maximum,DeepHit/gradients/DeepHit/Mean_grad/Prod_1_1/DeepHit/gradients/DeepHit/Mean_grad/Maximum/y_1*
T0
Ю
.DeepHit/gradients/DeepHit/Mean_grad/floordiv_1FloorDiv*DeepHit/gradients/DeepHit/Mean_grad/Prod_2-DeepHit/gradients/DeepHit/Mean_grad/Maximum_1*
T0
К
*DeepHit/gradients/DeepHit/Mean_grad/Cast_1Cast.DeepHit/gradients/DeepHit/Mean_grad/floordiv_1*

DstT0*

SrcT0*
Truncate( 
Щ
-DeepHit/gradients/DeepHit/Mean_grad/truediv_1RealDiv*DeepHit/gradients/DeepHit/Mean_grad/Tile_1*DeepHit/gradients/DeepHit/Mean_grad/Cast_1*
T0
d
-DeepHit/gradients/DeepHit/Mean_3_grad/Shape_3ShapeDeepHit/Reshape_6_1*
T0*
out_type0
≥
3DeepHit/gradients/DeepHit/Mean_3_grad/BroadcastTo_1BroadcastTo+DeepHit/gradients/DeepHit/Sum_4_grad/Tile_1-DeepHit/gradients/DeepHit/Mean_3_grad/Shape_3*
T0*

Tidx0
f
/DeepHit/gradients/DeepHit/Mean_3_grad/Shape_1_1ShapeDeepHit/Reshape_6_1*
T0*
out_type0
c
/DeepHit/gradients/DeepHit/Mean_3_grad/Shape_2_1ShapeDeepHit/Mean_3_1*
T0*
out_type0
[
-DeepHit/gradients/DeepHit/Mean_3_grad/Const_2Const*
dtype0*
valueB: 
Ї
,DeepHit/gradients/DeepHit/Mean_3_grad/Prod_2Prod/DeepHit/gradients/DeepHit/Mean_3_grad/Shape_1_1-DeepHit/gradients/DeepHit/Mean_3_grad/Const_2*
T0*

Tidx0*
	keep_dims( 
]
/DeepHit/gradients/DeepHit/Mean_3_grad/Const_1_1Const*
dtype0*
valueB: 
Њ
.DeepHit/gradients/DeepHit/Mean_3_grad/Prod_1_1Prod/DeepHit/gradients/DeepHit/Mean_3_grad/Shape_2_1/DeepHit/gradients/DeepHit/Mean_3_grad/Const_1_1*
T0*

Tidx0*
	keep_dims( 
[
1DeepHit/gradients/DeepHit/Mean_3_grad/Maximum/y_1Const*
dtype0*
value	B :
¶
/DeepHit/gradients/DeepHit/Mean_3_grad/Maximum_1Maximum.DeepHit/gradients/DeepHit/Mean_3_grad/Prod_1_11DeepHit/gradients/DeepHit/Mean_3_grad/Maximum/y_1*
T0
§
0DeepHit/gradients/DeepHit/Mean_3_grad/floordiv_1FloorDiv,DeepHit/gradients/DeepHit/Mean_3_grad/Prod_2/DeepHit/gradients/DeepHit/Mean_3_grad/Maximum_1*
T0
О
,DeepHit/gradients/DeepHit/Mean_3_grad/Cast_1Cast0DeepHit/gradients/DeepHit/Mean_3_grad/floordiv_1*

DstT0*

SrcT0*
Truncate( 
¶
/DeepHit/gradients/DeepHit/Mean_3_grad/truediv_1RealDiv3DeepHit/gradients/DeepHit/Mean_3_grad/BroadcastTo_1,DeepHit/gradients/DeepHit/Mean_3_grad/Cast_1*
T0
e
0DeepHit/gradients/DeepHit/Reshape_9_grad/Shape_1ShapeDeepHit/stack_2_1*
T0*
out_type0
Ј
2DeepHit/gradients/DeepHit/Reshape_9_grad/Reshape_1Reshape/DeepHit/gradients/DeepHit/Mean_6_grad/truediv_10DeepHit/gradients/DeepHit/Reshape_9_grad/Shape_1*
T0*
Tshape0
_
,DeepHit/gradients/DeepHit/add_2_grad/Shape_2ShapeDeepHit/mul_1_1*
T0*
out_type0
a
.DeepHit/gradients/DeepHit/add_2_grad/Shape_1_1ShapeDeepHit/mul_4_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/add_2_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/add_2_grad/Shape_2.DeepHit/gradients/DeepHit/add_2_grad/Shape_1_1*
T0
ƒ
*DeepHit/gradients/DeepHit/add_2_grad/Sum_2Sum-DeepHit/gradients/DeepHit/Mean_grad/truediv_1<DeepHit/gradients/DeepHit/add_2_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/add_2_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/add_2_grad/Sum_2,DeepHit/gradients/DeepHit/add_2_grad/Shape_2*
T0*
Tshape0
»
,DeepHit/gradients/DeepHit/add_2_grad/Sum_1_1Sum-DeepHit/gradients/DeepHit/Mean_grad/truediv_1>DeepHit/gradients/DeepHit/add_2_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/add_2_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/add_2_grad/Sum_1_1.DeepHit/gradients/DeepHit/add_2_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/add_2_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/add_2_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/add_2_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/add_2_grad/Reshape_28^DeepHit/gradients/DeepHit/add_2_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/add_2_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/add_2_grad/Reshape_1_18^DeepHit/gradients/DeepHit/add_2_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/add_2_grad/Reshape_1_1
e
0DeepHit/gradients/DeepHit/Reshape_6_grad/Shape_1ShapeDeepHit/stack_1_1*
T0*
out_type0
Ј
2DeepHit/gradients/DeepHit/Reshape_6_grad/Reshape_1Reshape/DeepHit/gradients/DeepHit/Mean_3_grad/truediv_10DeepHit/gradients/DeepHit/Reshape_6_grad/Shape_1*
T0*
Tshape0
О
0DeepHit/gradients/DeepHit/stack_2_grad/unstack_1Unpack2DeepHit/gradients/DeepHit/Reshape_9_grad/Reshape_1*
T0*

axis*	
num
t
9DeepHit/gradients/DeepHit/stack_2_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/stack_2_grad/unstack_1
Й
ADeepHit/gradients/DeepHit/stack_2_grad/tuple/control_dependency_2Identity0DeepHit/gradients/DeepHit/stack_2_grad/unstack_1:^DeepHit/gradients/DeepHit/stack_2_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/stack_2_grad/unstack_1
Н
CDeepHit/gradients/DeepHit/stack_2_grad/tuple/control_dependency_1_1Identity2DeepHit/gradients/DeepHit/stack_2_grad/unstack_1:1:^DeepHit/gradients/DeepHit/stack_2_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/stack_2_grad/unstack_1
^
,DeepHit/gradients/DeepHit/mul_1_grad/Shape_2ShapeDeepHit/Sign_3*
T0*
out_type0
_
.DeepHit/gradients/DeepHit/mul_1_grad/Shape_1_1ShapeDeepHit/Log_2*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/mul_1_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/mul_1_grad/Shape_2.DeepHit/gradients/DeepHit/mul_1_grad/Shape_1_1*
T0
К
*DeepHit/gradients/DeepHit/mul_1_grad/Mul_2Mul?DeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependency_2DeepHit/Log_2*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_1_grad/Sum_2Sum*DeepHit/gradients/DeepHit/mul_1_grad/Mul_2<DeepHit/gradients/DeepHit/mul_1_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_1_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/mul_1_grad/Sum_2,DeepHit/gradients/DeepHit/mul_1_grad/Shape_2*
T0*
Tshape0
Н
,DeepHit/gradients/DeepHit/mul_1_grad/Mul_1_1MulDeepHit/Sign_3?DeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependency_2*
T0
«
,DeepHit/gradients/DeepHit/mul_1_grad/Sum_1_1Sum,DeepHit/gradients/DeepHit/mul_1_grad/Mul_1_1>DeepHit/gradients/DeepHit/mul_1_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/mul_1_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/mul_1_grad/Sum_1_1.DeepHit/gradients/DeepHit/mul_1_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/mul_1_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/mul_1_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/mul_1_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/mul_1_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/mul_1_grad/Reshape_28^DeepHit/gradients/DeepHit/mul_1_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_1_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/mul_1_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/mul_1_grad/Reshape_1_18^DeepHit/gradients/DeepHit/mul_1_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/mul_1_grad/Reshape_1_1
a
,DeepHit/gradients/DeepHit/mul_4_grad/Shape_2ShapeDeepHit/mul_4/x_1*
T0*
out_type0
a
.DeepHit/gradients/DeepHit/mul_4_grad/Shape_1_1ShapeDeepHit/mul_3_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/mul_4_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/mul_4_grad/Shape_2.DeepHit/gradients/DeepHit/mul_4_grad/Shape_1_1*
T0
О
*DeepHit/gradients/DeepHit/mul_4_grad/Mul_2MulADeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependency_1_1DeepHit/mul_3_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_4_grad/Sum_2Sum*DeepHit/gradients/DeepHit/mul_4_grad/Mul_2<DeepHit/gradients/DeepHit/mul_4_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_4_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/mul_4_grad/Sum_2,DeepHit/gradients/DeepHit/mul_4_grad/Shape_2*
T0*
Tshape0
Т
,DeepHit/gradients/DeepHit/mul_4_grad/Mul_1_1MulDeepHit/mul_4/x_1ADeepHit/gradients/DeepHit/add_2_grad/tuple/control_dependency_1_1*
T0
«
,DeepHit/gradients/DeepHit/mul_4_grad/Sum_1_1Sum,DeepHit/gradients/DeepHit/mul_4_grad/Mul_1_1>DeepHit/gradients/DeepHit/mul_4_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/mul_4_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/mul_4_grad/Sum_1_1.DeepHit/gradients/DeepHit/mul_4_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/mul_4_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/mul_4_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/mul_4_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/mul_4_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/mul_4_grad/Reshape_28^DeepHit/gradients/DeepHit/mul_4_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_4_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/mul_4_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/mul_4_grad/Reshape_1_18^DeepHit/gradients/DeepHit/mul_4_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/mul_4_grad/Reshape_1_1
О
0DeepHit/gradients/DeepHit/stack_1_grad/unstack_1Unpack2DeepHit/gradients/DeepHit/Reshape_6_grad/Reshape_1*
T0*

axis*	
num
t
9DeepHit/gradients/DeepHit/stack_1_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/stack_1_grad/unstack_1
Й
ADeepHit/gradients/DeepHit/stack_1_grad/tuple/control_dependency_2Identity0DeepHit/gradients/DeepHit/stack_1_grad/unstack_1:^DeepHit/gradients/DeepHit/stack_1_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/stack_1_grad/unstack_1
Н
CDeepHit/gradients/DeepHit/stack_1_grad/tuple/control_dependency_1_1Identity2DeepHit/gradients/DeepHit/stack_1_grad/unstack_1:1:^DeepHit/gradients/DeepHit/stack_1_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/stack_1_grad/unstack_1
^
-DeepHit/gradients/DeepHit/Mean_4_grad/Shape_3ShapeDeepHit/pow_2*
T0*
out_type0
…
3DeepHit/gradients/DeepHit/Mean_4_grad/BroadcastTo_1BroadcastToADeepHit/gradients/DeepHit/stack_2_grad/tuple/control_dependency_2-DeepHit/gradients/DeepHit/Mean_4_grad/Shape_3*
T0*

Tidx0
`
/DeepHit/gradients/DeepHit/Mean_4_grad/Shape_1_1ShapeDeepHit/pow_2*
T0*
out_type0
c
/DeepHit/gradients/DeepHit/Mean_4_grad/Shape_2_1ShapeDeepHit/Mean_4_1*
T0*
out_type0
[
-DeepHit/gradients/DeepHit/Mean_4_grad/Const_2Const*
dtype0*
valueB: 
Ї
,DeepHit/gradients/DeepHit/Mean_4_grad/Prod_2Prod/DeepHit/gradients/DeepHit/Mean_4_grad/Shape_1_1-DeepHit/gradients/DeepHit/Mean_4_grad/Const_2*
T0*

Tidx0*
	keep_dims( 
]
/DeepHit/gradients/DeepHit/Mean_4_grad/Const_1_1Const*
dtype0*
valueB: 
Њ
.DeepHit/gradients/DeepHit/Mean_4_grad/Prod_1_1Prod/DeepHit/gradients/DeepHit/Mean_4_grad/Shape_2_1/DeepHit/gradients/DeepHit/Mean_4_grad/Const_1_1*
T0*

Tidx0*
	keep_dims( 
[
1DeepHit/gradients/DeepHit/Mean_4_grad/Maximum/y_1Const*
dtype0*
value	B :
¶
/DeepHit/gradients/DeepHit/Mean_4_grad/Maximum_1Maximum.DeepHit/gradients/DeepHit/Mean_4_grad/Prod_1_11DeepHit/gradients/DeepHit/Mean_4_grad/Maximum/y_1*
T0
§
0DeepHit/gradients/DeepHit/Mean_4_grad/floordiv_1FloorDiv,DeepHit/gradients/DeepHit/Mean_4_grad/Prod_2/DeepHit/gradients/DeepHit/Mean_4_grad/Maximum_1*
T0
О
,DeepHit/gradients/DeepHit/Mean_4_grad/Cast_1Cast0DeepHit/gradients/DeepHit/Mean_4_grad/floordiv_1*

DstT0*

SrcT0*
Truncate( 
¶
/DeepHit/gradients/DeepHit/Mean_4_grad/truediv_1RealDiv3DeepHit/gradients/DeepHit/Mean_4_grad/BroadcastTo_1,DeepHit/gradients/DeepHit/Mean_4_grad/Cast_1*
T0
`
-DeepHit/gradients/DeepHit/Mean_5_grad/Shape_3ShapeDeepHit/pow_1_1*
T0*
out_type0
Ћ
3DeepHit/gradients/DeepHit/Mean_5_grad/BroadcastTo_1BroadcastToCDeepHit/gradients/DeepHit/stack_2_grad/tuple/control_dependency_1_1-DeepHit/gradients/DeepHit/Mean_5_grad/Shape_3*
T0*

Tidx0
b
/DeepHit/gradients/DeepHit/Mean_5_grad/Shape_1_1ShapeDeepHit/pow_1_1*
T0*
out_type0
c
/DeepHit/gradients/DeepHit/Mean_5_grad/Shape_2_1ShapeDeepHit/Mean_5_1*
T0*
out_type0
[
-DeepHit/gradients/DeepHit/Mean_5_grad/Const_2Const*
dtype0*
valueB: 
Ї
,DeepHit/gradients/DeepHit/Mean_5_grad/Prod_2Prod/DeepHit/gradients/DeepHit/Mean_5_grad/Shape_1_1-DeepHit/gradients/DeepHit/Mean_5_grad/Const_2*
T0*

Tidx0*
	keep_dims( 
]
/DeepHit/gradients/DeepHit/Mean_5_grad/Const_1_1Const*
dtype0*
valueB: 
Њ
.DeepHit/gradients/DeepHit/Mean_5_grad/Prod_1_1Prod/DeepHit/gradients/DeepHit/Mean_5_grad/Shape_2_1/DeepHit/gradients/DeepHit/Mean_5_grad/Const_1_1*
T0*

Tidx0*
	keep_dims( 
[
1DeepHit/gradients/DeepHit/Mean_5_grad/Maximum/y_1Const*
dtype0*
value	B :
¶
/DeepHit/gradients/DeepHit/Mean_5_grad/Maximum_1Maximum.DeepHit/gradients/DeepHit/Mean_5_grad/Prod_1_11DeepHit/gradients/DeepHit/Mean_5_grad/Maximum/y_1*
T0
§
0DeepHit/gradients/DeepHit/Mean_5_grad/floordiv_1FloorDiv,DeepHit/gradients/DeepHit/Mean_5_grad/Prod_2/DeepHit/gradients/DeepHit/Mean_5_grad/Maximum_1*
T0
О
,DeepHit/gradients/DeepHit/Mean_5_grad/Cast_1Cast0DeepHit/gradients/DeepHit/Mean_5_grad/floordiv_1*

DstT0*

SrcT0*
Truncate( 
¶
/DeepHit/gradients/DeepHit/Mean_5_grad/truediv_1RealDiv3DeepHit/gradients/DeepHit/Mean_5_grad/BroadcastTo_1,DeepHit/gradients/DeepHit/Mean_5_grad/Cast_1*
T0
Щ
/DeepHit/gradients/DeepHit/Log_grad/Reciprocal_1
ReciprocalDeepHit/add_6B^DeepHit/gradients/DeepHit/mul_1_grad/tuple/control_dependency_1_1*
T0
ђ
(DeepHit/gradients/DeepHit/Log_grad/mul_1MulADeepHit/gradients/DeepHit/mul_1_grad/tuple/control_dependency_1_1/DeepHit/gradients/DeepHit/Log_grad/Reciprocal_1*
T0
_
,DeepHit/gradients/DeepHit/mul_3_grad/Shape_2ShapeDeepHit/sub_3_1*
T0*
out_type0
a
.DeepHit/gradients/DeepHit/mul_3_grad/Shape_1_1ShapeDeepHit/Log_1_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/mul_3_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/mul_3_grad/Shape_2.DeepHit/gradients/DeepHit/mul_3_grad/Shape_1_1*
T0
О
*DeepHit/gradients/DeepHit/mul_3_grad/Mul_2MulADeepHit/gradients/DeepHit/mul_4_grad/tuple/control_dependency_1_1DeepHit/Log_1_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_3_grad/Sum_2Sum*DeepHit/gradients/DeepHit/mul_3_grad/Mul_2<DeepHit/gradients/DeepHit/mul_3_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_3_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/mul_3_grad/Sum_2,DeepHit/gradients/DeepHit/mul_3_grad/Shape_2*
T0*
Tshape0
Р
,DeepHit/gradients/DeepHit/mul_3_grad/Mul_1_1MulDeepHit/sub_3_1ADeepHit/gradients/DeepHit/mul_4_grad/tuple/control_dependency_1_1*
T0
«
,DeepHit/gradients/DeepHit/mul_3_grad/Sum_1_1Sum,DeepHit/gradients/DeepHit/mul_3_grad/Mul_1_1>DeepHit/gradients/DeepHit/mul_3_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/mul_3_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/mul_3_grad/Sum_1_1.DeepHit/gradients/DeepHit/mul_3_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/mul_3_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/mul_3_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/mul_3_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/mul_3_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/mul_3_grad/Reshape_28^DeepHit/gradients/DeepHit/mul_3_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_3_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/mul_3_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/mul_3_grad/Reshape_1_18^DeepHit/gradients/DeepHit/mul_3_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/mul_3_grad/Reshape_1_1
`
-DeepHit/gradients/DeepHit/Mean_1_grad/Shape_3ShapeDeepHit/mul_5_1*
T0*
out_type0
…
3DeepHit/gradients/DeepHit/Mean_1_grad/BroadcastTo_1BroadcastToADeepHit/gradients/DeepHit/stack_1_grad/tuple/control_dependency_2-DeepHit/gradients/DeepHit/Mean_1_grad/Shape_3*
T0*

Tidx0
b
/DeepHit/gradients/DeepHit/Mean_1_grad/Shape_1_1ShapeDeepHit/mul_5_1*
T0*
out_type0
c
/DeepHit/gradients/DeepHit/Mean_1_grad/Shape_2_1ShapeDeepHit/Mean_1_1*
T0*
out_type0
[
-DeepHit/gradients/DeepHit/Mean_1_grad/Const_2Const*
dtype0*
valueB: 
Ї
,DeepHit/gradients/DeepHit/Mean_1_grad/Prod_2Prod/DeepHit/gradients/DeepHit/Mean_1_grad/Shape_1_1-DeepHit/gradients/DeepHit/Mean_1_grad/Const_2*
T0*

Tidx0*
	keep_dims( 
]
/DeepHit/gradients/DeepHit/Mean_1_grad/Const_1_1Const*
dtype0*
valueB: 
Њ
.DeepHit/gradients/DeepHit/Mean_1_grad/Prod_1_1Prod/DeepHit/gradients/DeepHit/Mean_1_grad/Shape_2_1/DeepHit/gradients/DeepHit/Mean_1_grad/Const_1_1*
T0*

Tidx0*
	keep_dims( 
[
1DeepHit/gradients/DeepHit/Mean_1_grad/Maximum/y_1Const*
dtype0*
value	B :
¶
/DeepHit/gradients/DeepHit/Mean_1_grad/Maximum_1Maximum.DeepHit/gradients/DeepHit/Mean_1_grad/Prod_1_11DeepHit/gradients/DeepHit/Mean_1_grad/Maximum/y_1*
T0
§
0DeepHit/gradients/DeepHit/Mean_1_grad/floordiv_1FloorDiv,DeepHit/gradients/DeepHit/Mean_1_grad/Prod_2/DeepHit/gradients/DeepHit/Mean_1_grad/Maximum_1*
T0
О
,DeepHit/gradients/DeepHit/Mean_1_grad/Cast_1Cast0DeepHit/gradients/DeepHit/Mean_1_grad/floordiv_1*

DstT0*

SrcT0*
Truncate( 
¶
/DeepHit/gradients/DeepHit/Mean_1_grad/truediv_1RealDiv3DeepHit/gradients/DeepHit/Mean_1_grad/BroadcastTo_1,DeepHit/gradients/DeepHit/Mean_1_grad/Cast_1*
T0
`
-DeepHit/gradients/DeepHit/Mean_2_grad/Shape_3ShapeDeepHit/mul_6_1*
T0*
out_type0
Ћ
3DeepHit/gradients/DeepHit/Mean_2_grad/BroadcastTo_1BroadcastToCDeepHit/gradients/DeepHit/stack_1_grad/tuple/control_dependency_1_1-DeepHit/gradients/DeepHit/Mean_2_grad/Shape_3*
T0*

Tidx0
b
/DeepHit/gradients/DeepHit/Mean_2_grad/Shape_1_1ShapeDeepHit/mul_6_1*
T0*
out_type0
c
/DeepHit/gradients/DeepHit/Mean_2_grad/Shape_2_1ShapeDeepHit/Mean_2_1*
T0*
out_type0
[
-DeepHit/gradients/DeepHit/Mean_2_grad/Const_2Const*
dtype0*
valueB: 
Ї
,DeepHit/gradients/DeepHit/Mean_2_grad/Prod_2Prod/DeepHit/gradients/DeepHit/Mean_2_grad/Shape_1_1-DeepHit/gradients/DeepHit/Mean_2_grad/Const_2*
T0*

Tidx0*
	keep_dims( 
]
/DeepHit/gradients/DeepHit/Mean_2_grad/Const_1_1Const*
dtype0*
valueB: 
Њ
.DeepHit/gradients/DeepHit/Mean_2_grad/Prod_1_1Prod/DeepHit/gradients/DeepHit/Mean_2_grad/Shape_2_1/DeepHit/gradients/DeepHit/Mean_2_grad/Const_1_1*
T0*

Tidx0*
	keep_dims( 
[
1DeepHit/gradients/DeepHit/Mean_2_grad/Maximum/y_1Const*
dtype0*
value	B :
¶
/DeepHit/gradients/DeepHit/Mean_2_grad/Maximum_1Maximum.DeepHit/gradients/DeepHit/Mean_2_grad/Prod_1_11DeepHit/gradients/DeepHit/Mean_2_grad/Maximum/y_1*
T0
§
0DeepHit/gradients/DeepHit/Mean_2_grad/floordiv_1FloorDiv,DeepHit/gradients/DeepHit/Mean_2_grad/Prod_2/DeepHit/gradients/DeepHit/Mean_2_grad/Maximum_1*
T0
О
,DeepHit/gradients/DeepHit/Mean_2_grad/Cast_1Cast0DeepHit/gradients/DeepHit/Mean_2_grad/floordiv_1*

DstT0*

SrcT0*
Truncate( 
¶
/DeepHit/gradients/DeepHit/Mean_2_grad/truediv_1RealDiv3DeepHit/gradients/DeepHit/Mean_2_grad/BroadcastTo_1,DeepHit/gradients/DeepHit/Mean_2_grad/Cast_1*
T0
]
*DeepHit/gradients/DeepHit/pow_grad/Shape_2ShapeDeepHit/sub_8_1*
T0*
out_type0
_
,DeepHit/gradients/DeepHit/pow_grad/Shape_1_1ShapeDeepHit/pow/y_1*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/pow_grad/BroadcastGradientArgs_1BroadcastGradientArgs*DeepHit/gradients/DeepHit/pow_grad/Shape_2,DeepHit/gradients/DeepHit/pow_grad/Shape_1_1*
T0
z
(DeepHit/gradients/DeepHit/pow_grad/mul_4Mul/DeepHit/gradients/DeepHit/Mean_4_grad/truediv_1DeepHit/pow/y_1*
T0
W
*DeepHit/gradients/DeepHit/pow_grad/sub/y_1Const*
dtype0*
valueB
 *  А?
u
(DeepHit/gradients/DeepHit/pow_grad/sub_1SubDeepHit/pow/y_1*DeepHit/gradients/DeepHit/pow_grad/sub/y_1*
T0
s
(DeepHit/gradients/DeepHit/pow_grad/Pow_1PowDeepHit/sub_8_1(DeepHit/gradients/DeepHit/pow_grad/sub_1*
T0
О
*DeepHit/gradients/DeepHit/pow_grad/mul_1_1Mul(DeepHit/gradients/DeepHit/pow_grad/mul_4(DeepHit/gradients/DeepHit/pow_grad/Pow_1*
T0
љ
(DeepHit/gradients/DeepHit/pow_grad/Sum_2Sum*DeepHit/gradients/DeepHit/pow_grad/mul_1_1:DeepHit/gradients/DeepHit/pow_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/pow_grad/Reshape_2Reshape(DeepHit/gradients/DeepHit/pow_grad/Sum_2*DeepHit/gradients/DeepHit/pow_grad/Shape_2*
T0*
Tshape0
[
.DeepHit/gradients/DeepHit/pow_grad/Greater/y_1Const*
dtype0*
valueB
 *    
Б
,DeepHit/gradients/DeepHit/pow_grad/Greater_1GreaterDeepHit/sub_8_1.DeepHit/gradients/DeepHit/pow_grad/Greater/y_1*
T0
g
4DeepHit/gradients/DeepHit/pow_grad/ones_like/Shape_1ShapeDeepHit/sub_8_1*
T0*
out_type0
a
4DeepHit/gradients/DeepHit/pow_grad/ones_like/Const_1Const*
dtype0*
valueB
 *  А?
љ
.DeepHit/gradients/DeepHit/pow_grad/ones_like_1Fill4DeepHit/gradients/DeepHit/pow_grad/ones_like/Shape_14DeepHit/gradients/DeepHit/pow_grad/ones_like/Const_1*
T0*

index_type0
≠
+DeepHit/gradients/DeepHit/pow_grad/Select_2Select,DeepHit/gradients/DeepHit/pow_grad/Greater_1DeepHit/sub_8_1.DeepHit/gradients/DeepHit/pow_grad/ones_like_1*
T0
e
(DeepHit/gradients/DeepHit/pow_grad/Log_1Log+DeepHit/gradients/DeepHit/pow_grad/Select_2*
T0
V
/DeepHit/gradients/DeepHit/pow_grad/zeros_like_1	ZerosLikeDeepHit/sub_8_1*
T0
…
-DeepHit/gradients/DeepHit/pow_grad/Select_1_1Select,DeepHit/gradients/DeepHit/pow_grad/Greater_1(DeepHit/gradients/DeepHit/pow_grad/Log_1/DeepHit/gradients/DeepHit/pow_grad/zeros_like_1*
T0
z
*DeepHit/gradients/DeepHit/pow_grad/mul_2_1Mul/DeepHit/gradients/DeepHit/Mean_4_grad/truediv_1DeepHit/pow_2*
T0
Х
*DeepHit/gradients/DeepHit/pow_grad/mul_3_1Mul*DeepHit/gradients/DeepHit/pow_grad/mul_2_1-DeepHit/gradients/DeepHit/pow_grad/Select_1_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/pow_grad/Sum_1_1Sum*DeepHit/gradients/DeepHit/pow_grad/mul_3_1<DeepHit/gradients/DeepHit/pow_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/pow_grad/Reshape_1_1Reshape*DeepHit/gradients/DeepHit/pow_grad/Sum_1_1,DeepHit/gradients/DeepHit/pow_grad/Shape_1_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/pow_grad/tuple/group_deps_1NoOp/^DeepHit/gradients/DeepHit/pow_grad/Reshape_1_1-^DeepHit/gradients/DeepHit/pow_grad/Reshape_2
щ
=DeepHit/gradients/DeepHit/pow_grad/tuple/control_dependency_2Identity,DeepHit/gradients/DeepHit/pow_grad/Reshape_26^DeepHit/gradients/DeepHit/pow_grad/tuple/group_deps_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/pow_grad/Reshape_2
€
?DeepHit/gradients/DeepHit/pow_grad/tuple/control_dependency_1_1Identity.DeepHit/gradients/DeepHit/pow_grad/Reshape_1_16^DeepHit/gradients/DeepHit/pow_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/pow_grad/Reshape_1_1
_
,DeepHit/gradients/DeepHit/pow_1_grad/Shape_2ShapeDeepHit/sub_9_1*
T0*
out_type0
c
.DeepHit/gradients/DeepHit/pow_1_grad/Shape_1_1ShapeDeepHit/pow_1/y_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/pow_1_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/pow_1_grad/Shape_2.DeepHit/gradients/DeepHit/pow_1_grad/Shape_1_1*
T0
~
*DeepHit/gradients/DeepHit/pow_1_grad/mul_4Mul/DeepHit/gradients/DeepHit/Mean_5_grad/truediv_1DeepHit/pow_1/y_1*
T0
Y
,DeepHit/gradients/DeepHit/pow_1_grad/sub/y_1Const*
dtype0*
valueB
 *  А?
{
*DeepHit/gradients/DeepHit/pow_1_grad/sub_1SubDeepHit/pow_1/y_1,DeepHit/gradients/DeepHit/pow_1_grad/sub/y_1*
T0
w
*DeepHit/gradients/DeepHit/pow_1_grad/Pow_1PowDeepHit/sub_9_1*DeepHit/gradients/DeepHit/pow_1_grad/sub_1*
T0
Ф
,DeepHit/gradients/DeepHit/pow_1_grad/mul_1_1Mul*DeepHit/gradients/DeepHit/pow_1_grad/mul_4*DeepHit/gradients/DeepHit/pow_1_grad/Pow_1*
T0
√
*DeepHit/gradients/DeepHit/pow_1_grad/Sum_2Sum,DeepHit/gradients/DeepHit/pow_1_grad/mul_1_1<DeepHit/gradients/DeepHit/pow_1_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/pow_1_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/pow_1_grad/Sum_2,DeepHit/gradients/DeepHit/pow_1_grad/Shape_2*
T0*
Tshape0
]
0DeepHit/gradients/DeepHit/pow_1_grad/Greater/y_1Const*
dtype0*
valueB
 *    
Е
.DeepHit/gradients/DeepHit/pow_1_grad/Greater_1GreaterDeepHit/sub_9_10DeepHit/gradients/DeepHit/pow_1_grad/Greater/y_1*
T0
i
6DeepHit/gradients/DeepHit/pow_1_grad/ones_like/Shape_1ShapeDeepHit/sub_9_1*
T0*
out_type0
c
6DeepHit/gradients/DeepHit/pow_1_grad/ones_like/Const_1Const*
dtype0*
valueB
 *  А?
√
0DeepHit/gradients/DeepHit/pow_1_grad/ones_like_1Fill6DeepHit/gradients/DeepHit/pow_1_grad/ones_like/Shape_16DeepHit/gradients/DeepHit/pow_1_grad/ones_like/Const_1*
T0*

index_type0
≥
-DeepHit/gradients/DeepHit/pow_1_grad/Select_2Select.DeepHit/gradients/DeepHit/pow_1_grad/Greater_1DeepHit/sub_9_10DeepHit/gradients/DeepHit/pow_1_grad/ones_like_1*
T0
i
*DeepHit/gradients/DeepHit/pow_1_grad/Log_1Log-DeepHit/gradients/DeepHit/pow_1_grad/Select_2*
T0
X
1DeepHit/gradients/DeepHit/pow_1_grad/zeros_like_1	ZerosLikeDeepHit/sub_9_1*
T0
—
/DeepHit/gradients/DeepHit/pow_1_grad/Select_1_1Select.DeepHit/gradients/DeepHit/pow_1_grad/Greater_1*DeepHit/gradients/DeepHit/pow_1_grad/Log_11DeepHit/gradients/DeepHit/pow_1_grad/zeros_like_1*
T0
~
,DeepHit/gradients/DeepHit/pow_1_grad/mul_2_1Mul/DeepHit/gradients/DeepHit/Mean_5_grad/truediv_1DeepHit/pow_1_1*
T0
Ы
,DeepHit/gradients/DeepHit/pow_1_grad/mul_3_1Mul,DeepHit/gradients/DeepHit/pow_1_grad/mul_2_1/DeepHit/gradients/DeepHit/pow_1_grad/Select_1_1*
T0
«
,DeepHit/gradients/DeepHit/pow_1_grad/Sum_1_1Sum,DeepHit/gradients/DeepHit/pow_1_grad/mul_3_1>DeepHit/gradients/DeepHit/pow_1_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/pow_1_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/pow_1_grad/Sum_1_1.DeepHit/gradients/DeepHit/pow_1_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/pow_1_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/pow_1_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/pow_1_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/pow_1_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/pow_1_grad/Reshape_28^DeepHit/gradients/DeepHit/pow_1_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/pow_1_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/pow_1_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/pow_1_grad/Reshape_1_18^DeepHit/gradients/DeepHit/pow_1_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/pow_1_grad/Reshape_1_1
]
*DeepHit/gradients/DeepHit/add_grad/Shape_2ShapeDeepHit/Sum_1_1*
T0*
out_type0
_
,DeepHit/gradients/DeepHit/add_grad/Shape_1_1ShapeDeepHit/add/y_1*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/add_grad/BroadcastGradientArgs_1BroadcastGradientArgs*DeepHit/gradients/DeepHit/add_grad/Shape_2,DeepHit/gradients/DeepHit/add_grad/Shape_1_1*
T0
ї
(DeepHit/gradients/DeepHit/add_grad/Sum_2Sum(DeepHit/gradients/DeepHit/Log_grad/mul_1:DeepHit/gradients/DeepHit/add_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/add_grad/Reshape_2Reshape(DeepHit/gradients/DeepHit/add_grad/Sum_2*DeepHit/gradients/DeepHit/add_grad/Shape_2*
T0*
Tshape0
њ
*DeepHit/gradients/DeepHit/add_grad/Sum_1_1Sum(DeepHit/gradients/DeepHit/Log_grad/mul_1<DeepHit/gradients/DeepHit/add_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/add_grad/Reshape_1_1Reshape*DeepHit/gradients/DeepHit/add_grad/Sum_1_1,DeepHit/gradients/DeepHit/add_grad/Shape_1_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/add_grad/tuple/group_deps_1NoOp/^DeepHit/gradients/DeepHit/add_grad/Reshape_1_1-^DeepHit/gradients/DeepHit/add_grad/Reshape_2
щ
=DeepHit/gradients/DeepHit/add_grad/tuple/control_dependency_2Identity,DeepHit/gradients/DeepHit/add_grad/Reshape_26^DeepHit/gradients/DeepHit/add_grad/tuple/group_deps_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/add_grad/Reshape_2
€
?DeepHit/gradients/DeepHit/add_grad/tuple/control_dependency_1_1Identity.DeepHit/gradients/DeepHit/add_grad/Reshape_1_16^DeepHit/gradients/DeepHit/add_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/add_grad/Reshape_1_1
Э
1DeepHit/gradients/DeepHit/Log_1_grad/Reciprocal_1
ReciprocalDeepHit/add_1_1B^DeepHit/gradients/DeepHit/mul_3_grad/tuple/control_dependency_1_1*
T0
∞
*DeepHit/gradients/DeepHit/Log_1_grad/mul_1MulADeepHit/gradients/DeepHit/mul_3_grad/tuple/control_dependency_1_11DeepHit/gradients/DeepHit/Log_1_grad/Reciprocal_1*
T0
b
,DeepHit/gradients/DeepHit/mul_5_grad/Shape_2ShapeDeepHit/MatMul_4_1*
T0*
out_type0
_
.DeepHit/gradients/DeepHit/mul_5_grad/Shape_1_1ShapeDeepHit/Exp_2*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/mul_5_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/mul_5_grad/Shape_2.DeepHit/gradients/DeepHit/mul_5_grad/Shape_1_1*
T0
z
*DeepHit/gradients/DeepHit/mul_5_grad/Mul_2Mul/DeepHit/gradients/DeepHit/Mean_1_grad/truediv_1DeepHit/Exp_2*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_5_grad/Sum_2Sum*DeepHit/gradients/DeepHit/mul_5_grad/Mul_2<DeepHit/gradients/DeepHit/mul_5_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_5_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/mul_5_grad/Sum_2,DeepHit/gradients/DeepHit/mul_5_grad/Shape_2*
T0*
Tshape0
Б
,DeepHit/gradients/DeepHit/mul_5_grad/Mul_1_1MulDeepHit/MatMul_4_1/DeepHit/gradients/DeepHit/Mean_1_grad/truediv_1*
T0
«
,DeepHit/gradients/DeepHit/mul_5_grad/Sum_1_1Sum,DeepHit/gradients/DeepHit/mul_5_grad/Mul_1_1>DeepHit/gradients/DeepHit/mul_5_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/mul_5_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/mul_5_grad/Sum_1_1.DeepHit/gradients/DeepHit/mul_5_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/mul_5_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/mul_5_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/mul_5_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/mul_5_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/mul_5_grad/Reshape_28^DeepHit/gradients/DeepHit/mul_5_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_5_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/mul_5_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/mul_5_grad/Reshape_1_18^DeepHit/gradients/DeepHit/mul_5_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/mul_5_grad/Reshape_1_1
b
,DeepHit/gradients/DeepHit/mul_6_grad/Shape_2ShapeDeepHit/MatMul_9_1*
T0*
out_type0
a
.DeepHit/gradients/DeepHit/mul_6_grad/Shape_1_1ShapeDeepHit/Exp_1_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/mul_6_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/mul_6_grad/Shape_2.DeepHit/gradients/DeepHit/mul_6_grad/Shape_1_1*
T0
|
*DeepHit/gradients/DeepHit/mul_6_grad/Mul_2Mul/DeepHit/gradients/DeepHit/Mean_2_grad/truediv_1DeepHit/Exp_1_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_6_grad/Sum_2Sum*DeepHit/gradients/DeepHit/mul_6_grad/Mul_2<DeepHit/gradients/DeepHit/mul_6_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_6_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/mul_6_grad/Sum_2,DeepHit/gradients/DeepHit/mul_6_grad/Shape_2*
T0*
Tshape0
Б
,DeepHit/gradients/DeepHit/mul_6_grad/Mul_1_1MulDeepHit/MatMul_9_1/DeepHit/gradients/DeepHit/Mean_2_grad/truediv_1*
T0
«
,DeepHit/gradients/DeepHit/mul_6_grad/Sum_1_1Sum,DeepHit/gradients/DeepHit/mul_6_grad/Mul_1_1>DeepHit/gradients/DeepHit/mul_6_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/mul_6_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/mul_6_grad/Sum_1_1.DeepHit/gradients/DeepHit/mul_6_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/mul_6_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/mul_6_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/mul_6_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/mul_6_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/mul_6_grad/Reshape_28^DeepHit/gradients/DeepHit/mul_6_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_6_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/mul_6_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/mul_6_grad/Reshape_1_18^DeepHit/gradients/DeepHit/mul_6_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/mul_6_grad/Reshape_1_1
_
,DeepHit/gradients/DeepHit/sub_8_grad/Shape_2ShapeDeepHit/Sum_5_1*
T0*
out_type0
b
.DeepHit/gradients/DeepHit/sub_8_grad/Shape_1_1ShapeDeepHit/Cast_2_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/sub_8_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/sub_8_grad/Shape_2.DeepHit/gradients/DeepHit/sub_8_grad/Shape_1_1*
T0
‘
*DeepHit/gradients/DeepHit/sub_8_grad/Sum_2Sum=DeepHit/gradients/DeepHit/pow_grad/tuple/control_dependency_2<DeepHit/gradients/DeepHit/sub_8_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/sub_8_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/sub_8_grad/Sum_2,DeepHit/gradients/DeepHit/sub_8_grad/Shape_2*
T0*
Tshape0
y
*DeepHit/gradients/DeepHit/sub_8_grad/Neg_1Neg=DeepHit/gradients/DeepHit/pow_grad/tuple/control_dependency_2*
T0
≈
,DeepHit/gradients/DeepHit/sub_8_grad/Sum_1_1Sum*DeepHit/gradients/DeepHit/sub_8_grad/Neg_1>DeepHit/gradients/DeepHit/sub_8_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/sub_8_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/sub_8_grad/Sum_1_1.DeepHit/gradients/DeepHit/sub_8_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/sub_8_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/sub_8_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/sub_8_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/sub_8_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/sub_8_grad/Reshape_28^DeepHit/gradients/DeepHit/sub_8_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/sub_8_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/sub_8_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/sub_8_grad/Reshape_1_18^DeepHit/gradients/DeepHit/sub_8_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/sub_8_grad/Reshape_1_1
_
,DeepHit/gradients/DeepHit/sub_9_grad/Shape_2ShapeDeepHit/Sum_6_1*
T0*
out_type0
b
.DeepHit/gradients/DeepHit/sub_9_grad/Shape_1_1ShapeDeepHit/Cast_3_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/sub_9_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/sub_9_grad/Shape_2.DeepHit/gradients/DeepHit/sub_9_grad/Shape_1_1*
T0
÷
*DeepHit/gradients/DeepHit/sub_9_grad/Sum_2Sum?DeepHit/gradients/DeepHit/pow_1_grad/tuple/control_dependency_2<DeepHit/gradients/DeepHit/sub_9_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/sub_9_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/sub_9_grad/Sum_2,DeepHit/gradients/DeepHit/sub_9_grad/Shape_2*
T0*
Tshape0
{
*DeepHit/gradients/DeepHit/sub_9_grad/Neg_1Neg?DeepHit/gradients/DeepHit/pow_1_grad/tuple/control_dependency_2*
T0
≈
,DeepHit/gradients/DeepHit/sub_9_grad/Sum_1_1Sum*DeepHit/gradients/DeepHit/sub_9_grad/Neg_1>DeepHit/gradients/DeepHit/sub_9_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/sub_9_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/sub_9_grad/Sum_1_1.DeepHit/gradients/DeepHit/sub_9_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/sub_9_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/sub_9_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/sub_9_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/sub_9_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/sub_9_grad/Reshape_28^DeepHit/gradients/DeepHit/sub_9_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/sub_9_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/sub_9_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/sub_9_grad/Reshape_1_18^DeepHit/gradients/DeepHit/sub_9_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/sub_9_grad/Reshape_1_1
]
,DeepHit/gradients/DeepHit/Sum_1_grad/Shape_1ShapeDeepHit/Sum_8*
T0*
out_type0
√
2DeepHit/gradients/DeepHit/Sum_1_grad/BroadcastTo_1BroadcastTo=DeepHit/gradients/DeepHit/add_grad/tuple/control_dependency_2,DeepHit/gradients/DeepHit/Sum_1_grad/Shape_1*
T0*

Tidx0
_
,DeepHit/gradients/DeepHit/add_1_grad/Shape_2ShapeDeepHit/Sum_3_1*
T0*
out_type0
c
.DeepHit/gradients/DeepHit/add_1_grad/Shape_1_1ShapeDeepHit/add_1/y_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/add_1_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/add_1_grad/Shape_2.DeepHit/gradients/DeepHit/add_1_grad/Shape_1_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/add_1_grad/Sum_2Sum*DeepHit/gradients/DeepHit/Log_1_grad/mul_1<DeepHit/gradients/DeepHit/add_1_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/add_1_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/add_1_grad/Sum_2,DeepHit/gradients/DeepHit/add_1_grad/Shape_2*
T0*
Tshape0
≈
,DeepHit/gradients/DeepHit/add_1_grad/Sum_1_1Sum*DeepHit/gradients/DeepHit/Log_1_grad/mul_1>DeepHit/gradients/DeepHit/add_1_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/add_1_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/add_1_grad/Sum_1_1.DeepHit/gradients/DeepHit/add_1_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/add_1_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/add_1_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/add_1_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/add_1_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/add_1_grad/Reshape_28^DeepHit/gradients/DeepHit/add_1_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/add_1_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/add_1_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/add_1_grad/Reshape_1_18^DeepHit/gradients/DeepHit/add_1_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/add_1_grad/Reshape_1_1
К
(DeepHit/gradients/DeepHit/Exp_grad/mul_1MulADeepHit/gradients/DeepHit/mul_5_grad/tuple/control_dependency_1_1DeepHit/Exp_2*
T0
О
*DeepHit/gradients/DeepHit/Exp_1_grad/mul_1MulADeepHit/gradients/DeepHit/mul_6_grad/tuple/control_dependency_1_1DeepHit/Exp_1_1*
T0
_
,DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2ShapeDeepHit/mul_7_1*
T0*
out_type0
Ц
+DeepHit/gradients/DeepHit/Sum_5_grad/Size_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2*
dtype0*
value	B :
Ќ
*DeepHit/gradients/DeepHit/Sum_5_grad/add_1AddV2!DeepHit/Sum_5/reduction_indices_1+DeepHit/gradients/DeepHit/Sum_5_grad/Size_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2
ў
*DeepHit/gradients/DeepHit/Sum_5_grad/mod_1FloorMod*DeepHit/gradients/DeepHit/Sum_5_grad/add_1+DeepHit/gradients/DeepHit/Sum_5_grad/Size_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2
Ш
.DeepHit/gradients/DeepHit/Sum_5_grad/Shape_1_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2*
dtype0*
valueB 
Э
2DeepHit/gradients/DeepHit/Sum_5_grad/range/start_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2*
dtype0*
value	B : 
Э
2DeepHit/gradients/DeepHit/Sum_5_grad/range/delta_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2*
dtype0*
value	B :
Ч
,DeepHit/gradients/DeepHit/Sum_5_grad/range_1Range2DeepHit/gradients/DeepHit/Sum_5_grad/range/start_1+DeepHit/gradients/DeepHit/Sum_5_grad/Size_12DeepHit/gradients/DeepHit/Sum_5_grad/range/delta_1*

Tidx0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2
Ь
1DeepHit/gradients/DeepHit/Sum_5_grad/ones/Const_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2*
dtype0*
value	B :
т
+DeepHit/gradients/DeepHit/Sum_5_grad/ones_1Fill.DeepHit/gradients/DeepHit/Sum_5_grad/Shape_1_11DeepHit/gradients/DeepHit/Sum_5_grad/ones/Const_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2*

index_type0
Ќ
4DeepHit/gradients/DeepHit/Sum_5_grad/DynamicStitch_1DynamicStitch,DeepHit/gradients/DeepHit/Sum_5_grad/range_1*DeepHit/gradients/DeepHit/Sum_5_grad/mod_1,DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2+DeepHit/gradients/DeepHit/Sum_5_grad/ones_1*
N*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2
«
.DeepHit/gradients/DeepHit/Sum_5_grad/Reshape_1Reshape?DeepHit/gradients/DeepHit/sub_8_grad/tuple/control_dependency_24DeepHit/gradients/DeepHit/Sum_5_grad/DynamicStitch_1*
T0*
Tshape0
і
2DeepHit/gradients/DeepHit/Sum_5_grad/BroadcastTo_1BroadcastTo.DeepHit/gradients/DeepHit/Sum_5_grad/Reshape_1,DeepHit/gradients/DeepHit/Sum_5_grad/Shape_2*
T0*

Tidx0
_
,DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2ShapeDeepHit/mul_8_1*
T0*
out_type0
Ц
+DeepHit/gradients/DeepHit/Sum_6_grad/Size_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2*
dtype0*
value	B :
Ќ
*DeepHit/gradients/DeepHit/Sum_6_grad/add_1AddV2!DeepHit/Sum_6/reduction_indices_1+DeepHit/gradients/DeepHit/Sum_6_grad/Size_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2
ў
*DeepHit/gradients/DeepHit/Sum_6_grad/mod_1FloorMod*DeepHit/gradients/DeepHit/Sum_6_grad/add_1+DeepHit/gradients/DeepHit/Sum_6_grad/Size_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2
Ш
.DeepHit/gradients/DeepHit/Sum_6_grad/Shape_1_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2*
dtype0*
valueB 
Э
2DeepHit/gradients/DeepHit/Sum_6_grad/range/start_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2*
dtype0*
value	B : 
Э
2DeepHit/gradients/DeepHit/Sum_6_grad/range/delta_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2*
dtype0*
value	B :
Ч
,DeepHit/gradients/DeepHit/Sum_6_grad/range_1Range2DeepHit/gradients/DeepHit/Sum_6_grad/range/start_1+DeepHit/gradients/DeepHit/Sum_6_grad/Size_12DeepHit/gradients/DeepHit/Sum_6_grad/range/delta_1*

Tidx0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2
Ь
1DeepHit/gradients/DeepHit/Sum_6_grad/ones/Const_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2*
dtype0*
value	B :
т
+DeepHit/gradients/DeepHit/Sum_6_grad/ones_1Fill.DeepHit/gradients/DeepHit/Sum_6_grad/Shape_1_11DeepHit/gradients/DeepHit/Sum_6_grad/ones/Const_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2*

index_type0
Ќ
4DeepHit/gradients/DeepHit/Sum_6_grad/DynamicStitch_1DynamicStitch,DeepHit/gradients/DeepHit/Sum_6_grad/range_1*DeepHit/gradients/DeepHit/Sum_6_grad/mod_1,DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2+DeepHit/gradients/DeepHit/Sum_6_grad/ones_1*
N*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2
«
.DeepHit/gradients/DeepHit/Sum_6_grad/Reshape_1Reshape?DeepHit/gradients/DeepHit/sub_9_grad/tuple/control_dependency_24DeepHit/gradients/DeepHit/Sum_6_grad/DynamicStitch_1*
T0*
Tshape0
і
2DeepHit/gradients/DeepHit/Sum_6_grad/BroadcastTo_1BroadcastTo.DeepHit/gradients/DeepHit/Sum_6_grad/Reshape_1,DeepHit/gradients/DeepHit/Sum_6_grad/Shape_2*
T0*

Tidx0
\
*DeepHit/gradients/DeepHit/Sum_grad/Shape_2ShapeDeepHit/mul_12*
T0*
out_type0
Т
)DeepHit/gradients/DeepHit/Sum_grad/Size_1Const*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape_2*
dtype0*
value	B :
≈
(DeepHit/gradients/DeepHit/Sum_grad/add_1AddV2DeepHit/Sum/reduction_indices_1)DeepHit/gradients/DeepHit/Sum_grad/Size_1*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape_2
—
(DeepHit/gradients/DeepHit/Sum_grad/mod_1FloorMod(DeepHit/gradients/DeepHit/Sum_grad/add_1)DeepHit/gradients/DeepHit/Sum_grad/Size_1*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape_2
Ф
,DeepHit/gradients/DeepHit/Sum_grad/Shape_1_1Const*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape_2*
dtype0*
valueB 
Щ
0DeepHit/gradients/DeepHit/Sum_grad/range/start_1Const*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape_2*
dtype0*
value	B : 
Щ
0DeepHit/gradients/DeepHit/Sum_grad/range/delta_1Const*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape_2*
dtype0*
value	B :
Н
*DeepHit/gradients/DeepHit/Sum_grad/range_1Range0DeepHit/gradients/DeepHit/Sum_grad/range/start_1)DeepHit/gradients/DeepHit/Sum_grad/Size_10DeepHit/gradients/DeepHit/Sum_grad/range/delta_1*

Tidx0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape_2
Ш
/DeepHit/gradients/DeepHit/Sum_grad/ones/Const_1Const*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape_2*
dtype0*
value	B :
к
)DeepHit/gradients/DeepHit/Sum_grad/ones_1Fill,DeepHit/gradients/DeepHit/Sum_grad/Shape_1_1/DeepHit/gradients/DeepHit/Sum_grad/ones/Const_1*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape_2*

index_type0
Ѕ
2DeepHit/gradients/DeepHit/Sum_grad/DynamicStitch_1DynamicStitch*DeepHit/gradients/DeepHit/Sum_grad/range_1(DeepHit/gradients/DeepHit/Sum_grad/mod_1*DeepHit/gradients/DeepHit/Sum_grad/Shape_2)DeepHit/gradients/DeepHit/Sum_grad/ones_1*
N*
T0*=
_class3
1/loc:@DeepHit/gradients/DeepHit/Sum_grad/Shape_2
ґ
,DeepHit/gradients/DeepHit/Sum_grad/Reshape_1Reshape2DeepHit/gradients/DeepHit/Sum_1_grad/BroadcastTo_12DeepHit/gradients/DeepHit/Sum_grad/DynamicStitch_1*
T0*
Tshape0
Ѓ
0DeepHit/gradients/DeepHit/Sum_grad/BroadcastTo_1BroadcastTo,DeepHit/gradients/DeepHit/Sum_grad/Reshape_1*DeepHit/gradients/DeepHit/Sum_grad/Shape_2*
T0*

Tidx0
_
,DeepHit/gradients/DeepHit/Sum_3_grad/Shape_1ShapeDeepHit/Sum_2_1*
T0*
out_type0
≈
2DeepHit/gradients/DeepHit/Sum_3_grad/BroadcastTo_1BroadcastTo?DeepHit/gradients/DeepHit/add_1_grad/tuple/control_dependency_2,DeepHit/gradients/DeepHit/Sum_3_grad/Shape_1*
T0*

Tidx0
a
.DeepHit/gradients/DeepHit/truediv_grad/Shape_2ShapeDeepHit/Neg_1_1*
T0*
out_type0
Y
0DeepHit/gradients/DeepHit/truediv_grad/Shape_1_1Const*
dtype0*
valueB 
¬
>DeepHit/gradients/DeepHit/truediv_grad/BroadcastGradientArgs_1BroadcastGradientArgs.DeepHit/gradients/DeepHit/truediv_grad/Shape_20DeepHit/gradients/DeepHit/truediv_grad/Shape_1_1*
T0
Б
0DeepHit/gradients/DeepHit/truediv_grad/RealDiv_3RealDiv(DeepHit/gradients/DeepHit/Exp_grad/mul_1DeepHit/Const_1_1*
T0
Ћ
,DeepHit/gradients/DeepHit/truediv_grad/Sum_2Sum0DeepHit/gradients/DeepHit/truediv_grad/RealDiv_3>DeepHit/gradients/DeepHit/truediv_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/truediv_grad/Reshape_2Reshape,DeepHit/gradients/DeepHit/truediv_grad/Sum_2.DeepHit/gradients/DeepHit/truediv_grad/Shape_2*
T0*
Tshape0
M
,DeepHit/gradients/DeepHit/truediv_grad/Neg_1NegDeepHit/Neg_1_1*
T0
З
2DeepHit/gradients/DeepHit/truediv_grad/RealDiv_1_1RealDiv,DeepHit/gradients/DeepHit/truediv_grad/Neg_1DeepHit/Const_1_1*
T0
Н
2DeepHit/gradients/DeepHit/truediv_grad/RealDiv_2_1RealDiv2DeepHit/gradients/DeepHit/truediv_grad/RealDiv_1_1DeepHit/Const_1_1*
T0
Ъ
,DeepHit/gradients/DeepHit/truediv_grad/mul_1Mul(DeepHit/gradients/DeepHit/Exp_grad/mul_12DeepHit/gradients/DeepHit/truediv_grad/RealDiv_2_1*
T0
Ћ
.DeepHit/gradients/DeepHit/truediv_grad/Sum_1_1Sum,DeepHit/gradients/DeepHit/truediv_grad/mul_1@DeepHit/gradients/DeepHit/truediv_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
ґ
2DeepHit/gradients/DeepHit/truediv_grad/Reshape_1_1Reshape.DeepHit/gradients/DeepHit/truediv_grad/Sum_1_10DeepHit/gradients/DeepHit/truediv_grad/Shape_1_1*
T0*
Tshape0
©
9DeepHit/gradients/DeepHit/truediv_grad/tuple/group_deps_1NoOp3^DeepHit/gradients/DeepHit/truediv_grad/Reshape_1_11^DeepHit/gradients/DeepHit/truediv_grad/Reshape_2
Й
ADeepHit/gradients/DeepHit/truediv_grad/tuple/control_dependency_2Identity0DeepHit/gradients/DeepHit/truediv_grad/Reshape_2:^DeepHit/gradients/DeepHit/truediv_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/truediv_grad/Reshape_2
П
CDeepHit/gradients/DeepHit/truediv_grad/tuple/control_dependency_1_1Identity2DeepHit/gradients/DeepHit/truediv_grad/Reshape_1_1:^DeepHit/gradients/DeepHit/truediv_grad/tuple/group_deps_1*
T0*E
_class;
97loc:@DeepHit/gradients/DeepHit/truediv_grad/Reshape_1_1
c
0DeepHit/gradients/DeepHit/truediv_1_grad/Shape_2ShapeDeepHit/Neg_2_1*
T0*
out_type0
[
2DeepHit/gradients/DeepHit/truediv_1_grad/Shape_1_1Const*
dtype0*
valueB 
»
@DeepHit/gradients/DeepHit/truediv_1_grad/BroadcastGradientArgs_1BroadcastGradientArgs0DeepHit/gradients/DeepHit/truediv_1_grad/Shape_22DeepHit/gradients/DeepHit/truediv_1_grad/Shape_1_1*
T0
Е
2DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv_3RealDiv*DeepHit/gradients/DeepHit/Exp_1_grad/mul_1DeepHit/Const_1_1*
T0
—
.DeepHit/gradients/DeepHit/truediv_1_grad/Sum_2Sum2DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv_3@DeepHit/gradients/DeepHit/truediv_1_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
ґ
2DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_2Reshape.DeepHit/gradients/DeepHit/truediv_1_grad/Sum_20DeepHit/gradients/DeepHit/truediv_1_grad/Shape_2*
T0*
Tshape0
O
.DeepHit/gradients/DeepHit/truediv_1_grad/Neg_1NegDeepHit/Neg_2_1*
T0
Л
4DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv_1_1RealDiv.DeepHit/gradients/DeepHit/truediv_1_grad/Neg_1DeepHit/Const_1_1*
T0
С
4DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv_2_1RealDiv4DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv_1_1DeepHit/Const_1_1*
T0
†
.DeepHit/gradients/DeepHit/truediv_1_grad/mul_1Mul*DeepHit/gradients/DeepHit/Exp_1_grad/mul_14DeepHit/gradients/DeepHit/truediv_1_grad/RealDiv_2_1*
T0
—
0DeepHit/gradients/DeepHit/truediv_1_grad/Sum_1_1Sum.DeepHit/gradients/DeepHit/truediv_1_grad/mul_1BDeepHit/gradients/DeepHit/truediv_1_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
Љ
4DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_1_1Reshape0DeepHit/gradients/DeepHit/truediv_1_grad/Sum_1_12DeepHit/gradients/DeepHit/truediv_1_grad/Shape_1_1*
T0*
Tshape0
ѓ
;DeepHit/gradients/DeepHit/truediv_1_grad/tuple/group_deps_1NoOp5^DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_1_13^DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_2
С
CDeepHit/gradients/DeepHit/truediv_1_grad/tuple/control_dependency_2Identity2DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_2<^DeepHit/gradients/DeepHit/truediv_1_grad/tuple/group_deps_1*
T0*E
_class;
97loc:@DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_2
Ч
EDeepHit/gradients/DeepHit/truediv_1_grad/tuple/control_dependency_1_1Identity4DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_1_1<^DeepHit/gradients/DeepHit/truediv_1_grad/tuple/group_deps_1*
T0*G
_class=
;9loc:@DeepHit/gradients/DeepHit/truediv_1_grad/Reshape_1_1
c
,DeepHit/gradients/DeepHit/mul_7_grad/Shape_2ShapeDeepHit/Reshape_7_1*
T0*
out_type0
a
.DeepHit/gradients/DeepHit/mul_7_grad/Shape_1_1ShapeDeepHit/mask2_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/mul_7_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/mul_7_grad/Shape_2.DeepHit/gradients/DeepHit/mul_7_grad/Shape_1_1*
T0

*DeepHit/gradients/DeepHit/mul_7_grad/Mul_2Mul2DeepHit/gradients/DeepHit/Sum_5_grad/BroadcastTo_1DeepHit/mask2_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_7_grad/Sum_2Sum*DeepHit/gradients/DeepHit/mul_7_grad/Mul_2<DeepHit/gradients/DeepHit/mul_7_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_7_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/mul_7_grad/Sum_2,DeepHit/gradients/DeepHit/mul_7_grad/Shape_2*
T0*
Tshape0
Е
,DeepHit/gradients/DeepHit/mul_7_grad/Mul_1_1MulDeepHit/Reshape_7_12DeepHit/gradients/DeepHit/Sum_5_grad/BroadcastTo_1*
T0
«
,DeepHit/gradients/DeepHit/mul_7_grad/Sum_1_1Sum,DeepHit/gradients/DeepHit/mul_7_grad/Mul_1_1>DeepHit/gradients/DeepHit/mul_7_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/mul_7_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/mul_7_grad/Sum_1_1.DeepHit/gradients/DeepHit/mul_7_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/mul_7_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/mul_7_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/mul_7_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/mul_7_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/mul_7_grad/Reshape_28^DeepHit/gradients/DeepHit/mul_7_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_7_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/mul_7_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/mul_7_grad/Reshape_1_18^DeepHit/gradients/DeepHit/mul_7_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/mul_7_grad/Reshape_1_1
c
,DeepHit/gradients/DeepHit/mul_8_grad/Shape_2ShapeDeepHit/Reshape_8_1*
T0*
out_type0
a
.DeepHit/gradients/DeepHit/mul_8_grad/Shape_1_1ShapeDeepHit/mask2_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/mul_8_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/mul_8_grad/Shape_2.DeepHit/gradients/DeepHit/mul_8_grad/Shape_1_1*
T0

*DeepHit/gradients/DeepHit/mul_8_grad/Mul_2Mul2DeepHit/gradients/DeepHit/Sum_6_grad/BroadcastTo_1DeepHit/mask2_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_8_grad/Sum_2Sum*DeepHit/gradients/DeepHit/mul_8_grad/Mul_2<DeepHit/gradients/DeepHit/mul_8_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_8_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/mul_8_grad/Sum_2,DeepHit/gradients/DeepHit/mul_8_grad/Shape_2*
T0*
Tshape0
Е
,DeepHit/gradients/DeepHit/mul_8_grad/Mul_1_1MulDeepHit/Reshape_8_12DeepHit/gradients/DeepHit/Sum_6_grad/BroadcastTo_1*
T0
«
,DeepHit/gradients/DeepHit/mul_8_grad/Sum_1_1Sum,DeepHit/gradients/DeepHit/mul_8_grad/Mul_1_1>DeepHit/gradients/DeepHit/mul_8_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/mul_8_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/mul_8_grad/Sum_1_1.DeepHit/gradients/DeepHit/mul_8_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/mul_8_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/mul_8_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/mul_8_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/mul_8_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/mul_8_grad/Reshape_28^DeepHit/gradients/DeepHit/mul_8_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_8_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/mul_8_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/mul_8_grad/Reshape_1_18^DeepHit/gradients/DeepHit/mul_8_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/mul_8_grad/Reshape_1_1
]
*DeepHit/gradients/DeepHit/mul_grad/Shape_2ShapeDeepHit/mask1_1*
T0*
out_type0
c
,DeepHit/gradients/DeepHit/mul_grad/Shape_1_1ShapeDeepHit/Reshape_1_1*
T0*
out_type0
ґ
:DeepHit/gradients/DeepHit/mul_grad/BroadcastGradientArgs_1BroadcastGradientArgs*DeepHit/gradients/DeepHit/mul_grad/Shape_2,DeepHit/gradients/DeepHit/mul_grad/Shape_1_1*
T0

(DeepHit/gradients/DeepHit/mul_grad/Mul_2Mul0DeepHit/gradients/DeepHit/Sum_grad/BroadcastTo_1DeepHit/Reshape_1_1*
T0
ї
(DeepHit/gradients/DeepHit/mul_grad/Sum_2Sum(DeepHit/gradients/DeepHit/mul_grad/Mul_2:DeepHit/gradients/DeepHit/mul_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
§
,DeepHit/gradients/DeepHit/mul_grad/Reshape_2Reshape(DeepHit/gradients/DeepHit/mul_grad/Sum_2*DeepHit/gradients/DeepHit/mul_grad/Shape_2*
T0*
Tshape0
}
*DeepHit/gradients/DeepHit/mul_grad/Mul_1_1MulDeepHit/mask1_10DeepHit/gradients/DeepHit/Sum_grad/BroadcastTo_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_grad/Sum_1_1Sum*DeepHit/gradients/DeepHit/mul_grad/Mul_1_1<DeepHit/gradients/DeepHit/mul_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_grad/Reshape_1_1Reshape*DeepHit/gradients/DeepHit/mul_grad/Sum_1_1,DeepHit/gradients/DeepHit/mul_grad/Shape_1_1*
T0*
Tshape0
Э
5DeepHit/gradients/DeepHit/mul_grad/tuple/group_deps_1NoOp/^DeepHit/gradients/DeepHit/mul_grad/Reshape_1_1-^DeepHit/gradients/DeepHit/mul_grad/Reshape_2
щ
=DeepHit/gradients/DeepHit/mul_grad/tuple/control_dependency_2Identity,DeepHit/gradients/DeepHit/mul_grad/Reshape_26^DeepHit/gradients/DeepHit/mul_grad/tuple/group_deps_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/mul_grad/Reshape_2
€
?DeepHit/gradients/DeepHit/mul_grad/tuple/control_dependency_1_1Identity.DeepHit/gradients/DeepHit/mul_grad/Reshape_1_16^DeepHit/gradients/DeepHit/mul_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_grad/Reshape_1_1
_
,DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2ShapeDeepHit/mul_2_1*
T0*
out_type0
Ц
+DeepHit/gradients/DeepHit/Sum_2_grad/Size_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2*
dtype0*
value	B :
Ќ
*DeepHit/gradients/DeepHit/Sum_2_grad/add_1AddV2!DeepHit/Sum_2/reduction_indices_1+DeepHit/gradients/DeepHit/Sum_2_grad/Size_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2
ў
*DeepHit/gradients/DeepHit/Sum_2_grad/mod_1FloorMod*DeepHit/gradients/DeepHit/Sum_2_grad/add_1+DeepHit/gradients/DeepHit/Sum_2_grad/Size_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2
Ш
.DeepHit/gradients/DeepHit/Sum_2_grad/Shape_1_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2*
dtype0*
valueB 
Э
2DeepHit/gradients/DeepHit/Sum_2_grad/range/start_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2*
dtype0*
value	B : 
Э
2DeepHit/gradients/DeepHit/Sum_2_grad/range/delta_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2*
dtype0*
value	B :
Ч
,DeepHit/gradients/DeepHit/Sum_2_grad/range_1Range2DeepHit/gradients/DeepHit/Sum_2_grad/range/start_1+DeepHit/gradients/DeepHit/Sum_2_grad/Size_12DeepHit/gradients/DeepHit/Sum_2_grad/range/delta_1*

Tidx0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2
Ь
1DeepHit/gradients/DeepHit/Sum_2_grad/ones/Const_1Const*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2*
dtype0*
value	B :
т
+DeepHit/gradients/DeepHit/Sum_2_grad/ones_1Fill.DeepHit/gradients/DeepHit/Sum_2_grad/Shape_1_11DeepHit/gradients/DeepHit/Sum_2_grad/ones/Const_1*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2*

index_type0
Ќ
4DeepHit/gradients/DeepHit/Sum_2_grad/DynamicStitch_1DynamicStitch,DeepHit/gradients/DeepHit/Sum_2_grad/range_1*DeepHit/gradients/DeepHit/Sum_2_grad/mod_1,DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2+DeepHit/gradients/DeepHit/Sum_2_grad/ones_1*
N*
T0*?
_class5
31loc:@DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2
Ї
.DeepHit/gradients/DeepHit/Sum_2_grad/Reshape_1Reshape2DeepHit/gradients/DeepHit/Sum_3_grad/BroadcastTo_14DeepHit/gradients/DeepHit/Sum_2_grad/DynamicStitch_1*
T0*
Tshape0
і
2DeepHit/gradients/DeepHit/Sum_2_grad/BroadcastTo_1BroadcastTo.DeepHit/gradients/DeepHit/Sum_2_grad/Reshape_1,DeepHit/gradients/DeepHit/Sum_2_grad/Shape_2*
T0*

Tidx0
}
*DeepHit/gradients/DeepHit/Neg_1_grad/Neg_1NegADeepHit/gradients/DeepHit/truediv_grad/tuple/control_dependency_2*
T0

*DeepHit/gradients/DeepHit/Neg_2_grad/Neg_1NegCDeepHit/gradients/DeepHit/truediv_1_grad/tuple/control_dependency_2*
T0
e
0DeepHit/gradients/DeepHit/Reshape_7_grad/Shape_1ShapeDeepHit/Slice_2_1*
T0*
out_type0
«
2DeepHit/gradients/DeepHit/Reshape_7_grad/Reshape_1Reshape?DeepHit/gradients/DeepHit/mul_7_grad/tuple/control_dependency_20DeepHit/gradients/DeepHit/Reshape_7_grad/Shape_1*
T0*
Tshape0
e
0DeepHit/gradients/DeepHit/Reshape_8_grad/Shape_1ShapeDeepHit/Slice_3_1*
T0*
out_type0
«
2DeepHit/gradients/DeepHit/Reshape_8_grad/Reshape_1Reshape?DeepHit/gradients/DeepHit/mul_8_grad/tuple/control_dependency_20DeepHit/gradients/DeepHit/Reshape_8_grad/Shape_1*
T0*
Tshape0
_
,DeepHit/gradients/DeepHit/mul_2_grad/Shape_2ShapeDeepHit/mask1_1*
T0*
out_type0
e
.DeepHit/gradients/DeepHit/mul_2_grad/Shape_1_1ShapeDeepHit/Reshape_1_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/mul_2_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/mul_2_grad/Shape_2.DeepHit/gradients/DeepHit/mul_2_grad/Shape_1_1*
T0
Г
*DeepHit/gradients/DeepHit/mul_2_grad/Mul_2Mul2DeepHit/gradients/DeepHit/Sum_2_grad/BroadcastTo_1DeepHit/Reshape_1_1*
T0
Ѕ
*DeepHit/gradients/DeepHit/mul_2_grad/Sum_2Sum*DeepHit/gradients/DeepHit/mul_2_grad/Mul_2<DeepHit/gradients/DeepHit/mul_2_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/mul_2_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/mul_2_grad/Sum_2,DeepHit/gradients/DeepHit/mul_2_grad/Shape_2*
T0*
Tshape0
Б
,DeepHit/gradients/DeepHit/mul_2_grad/Mul_1_1MulDeepHit/mask1_12DeepHit/gradients/DeepHit/Sum_2_grad/BroadcastTo_1*
T0
«
,DeepHit/gradients/DeepHit/mul_2_grad/Sum_1_1Sum,DeepHit/gradients/DeepHit/mul_2_grad/Mul_1_1>DeepHit/gradients/DeepHit/mul_2_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/mul_2_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/mul_2_grad/Sum_1_1.DeepHit/gradients/DeepHit/mul_2_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/mul_2_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/mul_2_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/mul_2_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/mul_2_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/mul_2_grad/Reshape_28^DeepHit/gradients/DeepHit/mul_2_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_2_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/mul_2_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/mul_2_grad/Reshape_1_18^DeepHit/gradients/DeepHit/mul_2_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/mul_2_grad/Reshape_1_1
x
>DeepHit/gradients/DeepHit/transpose_2_grad/InvertPermutation_1InvertPermutationDeepHit/transpose_2/perm_1*
T0
≈
6DeepHit/gradients/DeepHit/transpose_2_grad/transpose_1	Transpose*DeepHit/gradients/DeepHit/Neg_1_grad/Neg_1>DeepHit/gradients/DeepHit/transpose_2_grad/InvertPermutation_1*
T0*
Tperm0
x
>DeepHit/gradients/DeepHit/transpose_7_grad/InvertPermutation_1InvertPermutationDeepHit/transpose_7/perm_1*
T0
≈
6DeepHit/gradients/DeepHit/transpose_7_grad/transpose_1	Transpose*DeepHit/gradients/DeepHit/Neg_2_grad/Neg_1>DeepHit/gradients/DeepHit/transpose_7_grad/InvertPermutation_1*
T0*
Tperm0
W
-DeepHit/gradients/DeepHit/Slice_2_grad/Rank_1Const*
dtype0*
value	B :
c
.DeepHit/gradients/DeepHit/Slice_2_grad/Shape_2ShapeDeepHit/Slice_2_1*
T0*
out_type0
Z
0DeepHit/gradients/DeepHit/Slice_2_grad/stack/1_1Const*
dtype0*
value	B :
µ
.DeepHit/gradients/DeepHit/Slice_2_grad/stack_1Pack-DeepHit/gradients/DeepHit/Slice_2_grad/Rank_10DeepHit/gradients/DeepHit/Slice_2_grad/stack/1_1*
N*
T0*

axis 
Ы
0DeepHit/gradients/DeepHit/Slice_2_grad/Reshape_2ReshapeDeepHit/Slice_2/begin_1.DeepHit/gradients/DeepHit/Slice_2_grad/stack_1*
T0*
Tshape0
g
0DeepHit/gradients/DeepHit/Slice_2_grad/Shape_1_1ShapeDeepHit/Reshape_1_1*
T0*
out_type0
Ю
,DeepHit/gradients/DeepHit/Slice_2_grad/sub_2Sub0DeepHit/gradients/DeepHit/Slice_2_grad/Shape_1_1.DeepHit/gradients/DeepHit/Slice_2_grad/Shape_2*
T0
Е
.DeepHit/gradients/DeepHit/Slice_2_grad/sub_1_1Sub,DeepHit/gradients/DeepHit/Slice_2_grad/sub_2DeepHit/Slice_2/begin_1*
T0
і
2DeepHit/gradients/DeepHit/Slice_2_grad/Reshape_1_1Reshape.DeepHit/gradients/DeepHit/Slice_2_grad/sub_1_1.DeepHit/gradients/DeepHit/Slice_2_grad/stack_1*
T0*
Tshape0
^
4DeepHit/gradients/DeepHit/Slice_2_grad/concat/axis_1Const*
dtype0*
value	B :
х
/DeepHit/gradients/DeepHit/Slice_2_grad/concat_1ConcatV20DeepHit/gradients/DeepHit/Slice_2_grad/Reshape_22DeepHit/gradients/DeepHit/Slice_2_grad/Reshape_1_14DeepHit/gradients/DeepHit/Slice_2_grad/concat/axis_1*
N*
T0*

Tidx0
≤
,DeepHit/gradients/DeepHit/Slice_2_grad/Pad_1Pad2DeepHit/gradients/DeepHit/Reshape_7_grad/Reshape_1/DeepHit/gradients/DeepHit/Slice_2_grad/concat_1*
T0*
	Tpaddings0
W
-DeepHit/gradients/DeepHit/Slice_3_grad/Rank_1Const*
dtype0*
value	B :
c
.DeepHit/gradients/DeepHit/Slice_3_grad/Shape_2ShapeDeepHit/Slice_3_1*
T0*
out_type0
Z
0DeepHit/gradients/DeepHit/Slice_3_grad/stack/1_1Const*
dtype0*
value	B :
µ
.DeepHit/gradients/DeepHit/Slice_3_grad/stack_1Pack-DeepHit/gradients/DeepHit/Slice_3_grad/Rank_10DeepHit/gradients/DeepHit/Slice_3_grad/stack/1_1*
N*
T0*

axis 
Ы
0DeepHit/gradients/DeepHit/Slice_3_grad/Reshape_2ReshapeDeepHit/Slice_3/begin_1.DeepHit/gradients/DeepHit/Slice_3_grad/stack_1*
T0*
Tshape0
g
0DeepHit/gradients/DeepHit/Slice_3_grad/Shape_1_1ShapeDeepHit/Reshape_1_1*
T0*
out_type0
Ю
,DeepHit/gradients/DeepHit/Slice_3_grad/sub_2Sub0DeepHit/gradients/DeepHit/Slice_3_grad/Shape_1_1.DeepHit/gradients/DeepHit/Slice_3_grad/Shape_2*
T0
Е
.DeepHit/gradients/DeepHit/Slice_3_grad/sub_1_1Sub,DeepHit/gradients/DeepHit/Slice_3_grad/sub_2DeepHit/Slice_3/begin_1*
T0
і
2DeepHit/gradients/DeepHit/Slice_3_grad/Reshape_1_1Reshape.DeepHit/gradients/DeepHit/Slice_3_grad/sub_1_1.DeepHit/gradients/DeepHit/Slice_3_grad/stack_1*
T0*
Tshape0
^
4DeepHit/gradients/DeepHit/Slice_3_grad/concat/axis_1Const*
dtype0*
value	B :
х
/DeepHit/gradients/DeepHit/Slice_3_grad/concat_1ConcatV20DeepHit/gradients/DeepHit/Slice_3_grad/Reshape_22DeepHit/gradients/DeepHit/Slice_3_grad/Reshape_1_14DeepHit/gradients/DeepHit/Slice_3_grad/concat/axis_1*
N*
T0*

Tidx0
≤
,DeepHit/gradients/DeepHit/Slice_3_grad/Pad_1Pad2DeepHit/gradients/DeepHit/Reshape_8_grad/Reshape_1/DeepHit/gradients/DeepHit/Slice_3_grad/concat_1*
T0*
	Tpaddings0
b
,DeepHit/gradients/DeepHit/sub_4_grad/Shape_2ShapeDeepHit/MatMul_1_1*
T0*
out_type0
c
.DeepHit/gradients/DeepHit/sub_4_grad/Shape_1_1ShapeDeepHit/MatMul_10*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/sub_4_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/sub_4_grad/Shape_2.DeepHit/gradients/DeepHit/sub_4_grad/Shape_1_1*
T0
Ќ
*DeepHit/gradients/DeepHit/sub_4_grad/Sum_2Sum6DeepHit/gradients/DeepHit/transpose_2_grad/transpose_1<DeepHit/gradients/DeepHit/sub_4_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/sub_4_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/sub_4_grad/Sum_2,DeepHit/gradients/DeepHit/sub_4_grad/Shape_2*
T0*
Tshape0
r
*DeepHit/gradients/DeepHit/sub_4_grad/Neg_1Neg6DeepHit/gradients/DeepHit/transpose_2_grad/transpose_1*
T0
≈
,DeepHit/gradients/DeepHit/sub_4_grad/Sum_1_1Sum*DeepHit/gradients/DeepHit/sub_4_grad/Neg_1>DeepHit/gradients/DeepHit/sub_4_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/sub_4_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/sub_4_grad/Sum_1_1.DeepHit/gradients/DeepHit/sub_4_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/sub_4_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/sub_4_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/sub_4_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/sub_4_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/sub_4_grad/Reshape_28^DeepHit/gradients/DeepHit/sub_4_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/sub_4_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/sub_4_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/sub_4_grad/Reshape_1_18^DeepHit/gradients/DeepHit/sub_4_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/sub_4_grad/Reshape_1_1
b
,DeepHit/gradients/DeepHit/sub_6_grad/Shape_2ShapeDeepHit/MatMul_6_1*
T0*
out_type0
d
.DeepHit/gradients/DeepHit/sub_6_grad/Shape_1_1ShapeDeepHit/MatMul_5_1*
T0*
out_type0
Љ
<DeepHit/gradients/DeepHit/sub_6_grad/BroadcastGradientArgs_1BroadcastGradientArgs,DeepHit/gradients/DeepHit/sub_6_grad/Shape_2.DeepHit/gradients/DeepHit/sub_6_grad/Shape_1_1*
T0
Ќ
*DeepHit/gradients/DeepHit/sub_6_grad/Sum_2Sum6DeepHit/gradients/DeepHit/transpose_7_grad/transpose_1<DeepHit/gradients/DeepHit/sub_6_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
™
.DeepHit/gradients/DeepHit/sub_6_grad/Reshape_2Reshape*DeepHit/gradients/DeepHit/sub_6_grad/Sum_2,DeepHit/gradients/DeepHit/sub_6_grad/Shape_2*
T0*
Tshape0
r
*DeepHit/gradients/DeepHit/sub_6_grad/Neg_1Neg6DeepHit/gradients/DeepHit/transpose_7_grad/transpose_1*
T0
≈
,DeepHit/gradients/DeepHit/sub_6_grad/Sum_1_1Sum*DeepHit/gradients/DeepHit/sub_6_grad/Neg_1>DeepHit/gradients/DeepHit/sub_6_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
∞
0DeepHit/gradients/DeepHit/sub_6_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/sub_6_grad/Sum_1_1.DeepHit/gradients/DeepHit/sub_6_grad/Shape_1_1*
T0*
Tshape0
£
7DeepHit/gradients/DeepHit/sub_6_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/sub_6_grad/Reshape_1_1/^DeepHit/gradients/DeepHit/sub_6_grad/Reshape_2
Б
?DeepHit/gradients/DeepHit/sub_6_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/sub_6_grad/Reshape_28^DeepHit/gradients/DeepHit/sub_6_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/sub_6_grad/Reshape_2
З
ADeepHit/gradients/DeepHit/sub_6_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/sub_6_grad/Reshape_1_18^DeepHit/gradients/DeepHit/sub_6_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/sub_6_grad/Reshape_1_1
Ѕ
0DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_2MatMul?DeepHit/gradients/DeepHit/sub_4_grad/tuple/control_dependency_2DeepHit/transpose_1_1*
T0*
transpose_a( *
transpose_b(
Ѕ
2DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_1_1MatMulDeepHit/ones_like_4?DeepHit/gradients/DeepHit/sub_4_grad/tuple/control_dependency_2*
T0*
transpose_a(*
transpose_b( 
™
:DeepHit/gradients/DeepHit/MatMul_1_grad/tuple/group_deps_1NoOp3^DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_1_11^DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_2
Л
BDeepHit/gradients/DeepHit/MatMul_1_grad/tuple/control_dependency_2Identity0DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_2;^DeepHit/gradients/DeepHit/MatMul_1_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_2
С
DDeepHit/gradients/DeepHit/MatMul_1_grad/tuple/control_dependency_1_1Identity2DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_1_1;^DeepHit/gradients/DeepHit/MatMul_1_grad/tuple/group_deps_1*
T0*E
_class;
97loc:@DeepHit/gradients/DeepHit/MatMul_1_grad/MatMul_1_1
Ѕ
0DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_2MatMul?DeepHit/gradients/DeepHit/sub_6_grad/tuple/control_dependency_2DeepHit/transpose_6_1*
T0*
transpose_a( *
transpose_b(
√
2DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_1_1MatMulDeepHit/ones_like_1_1?DeepHit/gradients/DeepHit/sub_6_grad/tuple/control_dependency_2*
T0*
transpose_a(*
transpose_b( 
™
:DeepHit/gradients/DeepHit/MatMul_6_grad/tuple/group_deps_1NoOp3^DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_1_11^DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_2
Л
BDeepHit/gradients/DeepHit/MatMul_6_grad/tuple/control_dependency_2Identity0DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_2;^DeepHit/gradients/DeepHit/MatMul_6_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_2
С
DDeepHit/gradients/DeepHit/MatMul_6_grad/tuple/control_dependency_1_1Identity2DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_1_1;^DeepHit/gradients/DeepHit/MatMul_6_grad/tuple/group_deps_1*
T0*E
_class;
97loc:@DeepHit/gradients/DeepHit/MatMul_6_grad/MatMul_1_1
x
>DeepHit/gradients/DeepHit/transpose_1_grad/InvertPermutation_1InvertPermutationDeepHit/transpose_1/perm_1*
T0
я
6DeepHit/gradients/DeepHit/transpose_1_grad/transpose_1	TransposeDDeepHit/gradients/DeepHit/MatMul_1_grad/tuple/control_dependency_1_1>DeepHit/gradients/DeepHit/transpose_1_grad/InvertPermutation_1*
T0*
Tperm0
x
>DeepHit/gradients/DeepHit/transpose_6_grad/InvertPermutation_1InvertPermutationDeepHit/transpose_6/perm_1*
T0
я
6DeepHit/gradients/DeepHit/transpose_6_grad/transpose_1	TransposeDDeepHit/gradients/DeepHit/MatMul_6_grad/tuple/control_dependency_1_1>DeepHit/gradients/DeepHit/transpose_6_grad/InvertPermutation_1*
T0*
Tperm0
f
0DeepHit/gradients/DeepHit/Reshape_3_grad/Shape_1ShapeDeepHit/DiagPart_2*
T0*
out_type0
Њ
2DeepHit/gradients/DeepHit/Reshape_3_grad/Reshape_1Reshape6DeepHit/gradients/DeepHit/transpose_1_grad/transpose_10DeepHit/gradients/DeepHit/Reshape_3_grad/Shape_1*
T0*
Tshape0
h
0DeepHit/gradients/DeepHit/Reshape_5_grad/Shape_1ShapeDeepHit/DiagPart_1_1*
T0*
out_type0
Њ
2DeepHit/gradients/DeepHit/Reshape_5_grad/Reshape_1Reshape6DeepHit/gradients/DeepHit/transpose_6_grad/transpose_10DeepHit/gradients/DeepHit/Reshape_5_grad/Shape_1*
T0*
Tshape0
s
.DeepHit/gradients/DeepHit/DiagPart_grad/Diag_1Diag2DeepHit/gradients/DeepHit/Reshape_3_grad/Reshape_1*
T0
u
0DeepHit/gradients/DeepHit/DiagPart_1_grad/Diag_1Diag2DeepHit/gradients/DeepHit/Reshape_5_grad/Reshape_1*
T0
л
DeepHit/gradients/AddN_10AddNADeepHit/gradients/DeepHit/sub_4_grad/tuple/control_dependency_1_1.DeepHit/gradients/DeepHit/DiagPart_grad/Diag_1*
N*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/sub_4_grad/Reshape_1_1
Ш
.DeepHit/gradients/DeepHit/MatMul_grad/MatMul_2MatMulDeepHit/gradients/AddN_10DeepHit/transpose_10*
T0*
transpose_a( *
transpose_b(
Щ
0DeepHit/gradients/DeepHit/MatMul_grad/MatMul_1_1MatMulDeepHit/Reshape_2_1DeepHit/gradients/AddN_10*
T0*
transpose_a(*
transpose_b( 
§
8DeepHit/gradients/DeepHit/MatMul_grad/tuple/group_deps_1NoOp1^DeepHit/gradients/DeepHit/MatMul_grad/MatMul_1_1/^DeepHit/gradients/DeepHit/MatMul_grad/MatMul_2
Г
@DeepHit/gradients/DeepHit/MatMul_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/MatMul_grad/MatMul_29^DeepHit/gradients/DeepHit/MatMul_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/MatMul_grad/MatMul_2
Й
BDeepHit/gradients/DeepHit/MatMul_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/MatMul_grad/MatMul_1_19^DeepHit/gradients/DeepHit/MatMul_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/MatMul_grad/MatMul_1_1
о
DeepHit/gradients/AddN_1_1AddNADeepHit/gradients/DeepHit/sub_6_grad/tuple/control_dependency_1_10DeepHit/gradients/DeepHit/DiagPart_1_grad/Diag_1*
N*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/sub_6_grad/Reshape_1_1
Ь
0DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_2MatMulDeepHit/gradients/AddN_1_1DeepHit/transpose_5_1*
T0*
transpose_a( *
transpose_b(
Ь
2DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_1_1MatMulDeepHit/Reshape_4_1DeepHit/gradients/AddN_1_1*
T0*
transpose_a(*
transpose_b( 
™
:DeepHit/gradients/DeepHit/MatMul_5_grad/tuple/group_deps_1NoOp3^DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_1_11^DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_2
Л
BDeepHit/gradients/DeepHit/MatMul_5_grad/tuple/control_dependency_2Identity0DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_2;^DeepHit/gradients/DeepHit/MatMul_5_grad/tuple/group_deps_1*
T0*C
_class9
75loc:@DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_2
С
DDeepHit/gradients/DeepHit/MatMul_5_grad/tuple/control_dependency_1_1Identity2DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_1_1;^DeepHit/gradients/DeepHit/MatMul_5_grad/tuple/group_deps_1*
T0*E
_class;
97loc:@DeepHit/gradients/DeepHit/MatMul_5_grad/MatMul_1_1
c
0DeepHit/gradients/DeepHit/Reshape_2_grad/Shape_1ShapeDeepHit/Slice_4*
T0*
out_type0
»
2DeepHit/gradients/DeepHit/Reshape_2_grad/Reshape_1Reshape@DeepHit/gradients/DeepHit/MatMul_grad/tuple/control_dependency_20DeepHit/gradients/DeepHit/Reshape_2_grad/Shape_1*
T0*
Tshape0
e
0DeepHit/gradients/DeepHit/Reshape_4_grad/Shape_1ShapeDeepHit/Slice_1_1*
T0*
out_type0
 
2DeepHit/gradients/DeepHit/Reshape_4_grad/Reshape_1ReshapeBDeepHit/gradients/DeepHit/MatMul_5_grad/tuple/control_dependency_20DeepHit/gradients/DeepHit/Reshape_4_grad/Shape_1*
T0*
Tshape0
U
+DeepHit/gradients/DeepHit/Slice_grad/Rank_1Const*
dtype0*
value	B :
_
,DeepHit/gradients/DeepHit/Slice_grad/Shape_2ShapeDeepHit/Slice_4*
T0*
out_type0
X
.DeepHit/gradients/DeepHit/Slice_grad/stack/1_1Const*
dtype0*
value	B :
ѓ
,DeepHit/gradients/DeepHit/Slice_grad/stack_1Pack+DeepHit/gradients/DeepHit/Slice_grad/Rank_1.DeepHit/gradients/DeepHit/Slice_grad/stack/1_1*
N*
T0*

axis 
Х
.DeepHit/gradients/DeepHit/Slice_grad/Reshape_2ReshapeDeepHit/Slice/begin_1,DeepHit/gradients/DeepHit/Slice_grad/stack_1*
T0*
Tshape0
e
.DeepHit/gradients/DeepHit/Slice_grad/Shape_1_1ShapeDeepHit/Reshape_1_1*
T0*
out_type0
Ш
*DeepHit/gradients/DeepHit/Slice_grad/sub_2Sub.DeepHit/gradients/DeepHit/Slice_grad/Shape_1_1,DeepHit/gradients/DeepHit/Slice_grad/Shape_2*
T0

,DeepHit/gradients/DeepHit/Slice_grad/sub_1_1Sub*DeepHit/gradients/DeepHit/Slice_grad/sub_2DeepHit/Slice/begin_1*
T0
Ѓ
0DeepHit/gradients/DeepHit/Slice_grad/Reshape_1_1Reshape,DeepHit/gradients/DeepHit/Slice_grad/sub_1_1,DeepHit/gradients/DeepHit/Slice_grad/stack_1*
T0*
Tshape0
\
2DeepHit/gradients/DeepHit/Slice_grad/concat/axis_1Const*
dtype0*
value	B :
н
-DeepHit/gradients/DeepHit/Slice_grad/concat_1ConcatV2.DeepHit/gradients/DeepHit/Slice_grad/Reshape_20DeepHit/gradients/DeepHit/Slice_grad/Reshape_1_12DeepHit/gradients/DeepHit/Slice_grad/concat/axis_1*
N*
T0*

Tidx0
Ѓ
*DeepHit/gradients/DeepHit/Slice_grad/Pad_1Pad2DeepHit/gradients/DeepHit/Reshape_2_grad/Reshape_1-DeepHit/gradients/DeepHit/Slice_grad/concat_1*
T0*
	Tpaddings0
W
-DeepHit/gradients/DeepHit/Slice_1_grad/Rank_1Const*
dtype0*
value	B :
c
.DeepHit/gradients/DeepHit/Slice_1_grad/Shape_2ShapeDeepHit/Slice_1_1*
T0*
out_type0
Z
0DeepHit/gradients/DeepHit/Slice_1_grad/stack/1_1Const*
dtype0*
value	B :
µ
.DeepHit/gradients/DeepHit/Slice_1_grad/stack_1Pack-DeepHit/gradients/DeepHit/Slice_1_grad/Rank_10DeepHit/gradients/DeepHit/Slice_1_grad/stack/1_1*
N*
T0*

axis 
Ы
0DeepHit/gradients/DeepHit/Slice_1_grad/Reshape_2ReshapeDeepHit/Slice_1/begin_1.DeepHit/gradients/DeepHit/Slice_1_grad/stack_1*
T0*
Tshape0
g
0DeepHit/gradients/DeepHit/Slice_1_grad/Shape_1_1ShapeDeepHit/Reshape_1_1*
T0*
out_type0
Ю
,DeepHit/gradients/DeepHit/Slice_1_grad/sub_2Sub0DeepHit/gradients/DeepHit/Slice_1_grad/Shape_1_1.DeepHit/gradients/DeepHit/Slice_1_grad/Shape_2*
T0
Е
.DeepHit/gradients/DeepHit/Slice_1_grad/sub_1_1Sub,DeepHit/gradients/DeepHit/Slice_1_grad/sub_2DeepHit/Slice_1/begin_1*
T0
і
2DeepHit/gradients/DeepHit/Slice_1_grad/Reshape_1_1Reshape.DeepHit/gradients/DeepHit/Slice_1_grad/sub_1_1.DeepHit/gradients/DeepHit/Slice_1_grad/stack_1*
T0*
Tshape0
^
4DeepHit/gradients/DeepHit/Slice_1_grad/concat/axis_1Const*
dtype0*
value	B :
х
/DeepHit/gradients/DeepHit/Slice_1_grad/concat_1ConcatV20DeepHit/gradients/DeepHit/Slice_1_grad/Reshape_22DeepHit/gradients/DeepHit/Slice_1_grad/Reshape_1_14DeepHit/gradients/DeepHit/Slice_1_grad/concat/axis_1*
N*
T0*

Tidx0
≤
,DeepHit/gradients/DeepHit/Slice_1_grad/Pad_1Pad2DeepHit/gradients/DeepHit/Reshape_4_grad/Reshape_1/DeepHit/gradients/DeepHit/Slice_1_grad/concat_1*
T0*
	Tpaddings0
±
DeepHit/gradients/AddN_2_1AddN?DeepHit/gradients/DeepHit/mul_grad/tuple/control_dependency_1_1ADeepHit/gradients/DeepHit/mul_2_grad/tuple/control_dependency_1_1,DeepHit/gradients/DeepHit/Slice_2_grad/Pad_1,DeepHit/gradients/DeepHit/Slice_3_grad/Pad_1*DeepHit/gradients/DeepHit/Slice_grad/Pad_1,DeepHit/gradients/DeepHit/Slice_1_grad/Pad_1*
N*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/mul_grad/Reshape_1_1
l
0DeepHit/gradients/DeepHit/Reshape_1_grad/Shape_1ShapeDeepHit/Output/Softmax_1*
T0*
out_type0
Ґ
2DeepHit/gradients/DeepHit/Reshape_1_grad/Reshape_1ReshapeDeepHit/gradients/AddN_2_10DeepHit/gradients/DeepHit/Reshape_1_grad/Shape_1*
T0*
Tshape0
С
3DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_2Mul2DeepHit/gradients/DeepHit/Reshape_1_grad/Reshape_1DeepHit/Output/Softmax_1*
T0
x
EDeepHit/gradients/DeepHit/Output/Softmax_grad/Sum/reduction_indices_1Const*
dtype0*
valueB :
€€€€€€€€€
№
3DeepHit/gradients/DeepHit/Output/Softmax_grad/Sum_1Sum3DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_2EDeepHit/gradients/DeepHit/Output/Softmax_grad/Sum/reduction_indices_1*
T0*

Tidx0*
	keep_dims(
ђ
3DeepHit/gradients/DeepHit/Output/Softmax_grad/sub_1Sub2DeepHit/gradients/DeepHit/Reshape_1_grad/Reshape_13DeepHit/gradients/DeepHit/Output/Softmax_grad/Sum_1*
T0
Ф
5DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_1_1Mul3DeepHit/gradients/DeepHit/Output/Softmax_grad/sub_1DeepHit/Output/Softmax_1*
T0
°
;DeepHit/gradients/DeepHit/Output/BiasAdd_grad/BiasAddGrad_1BiasAddGrad5DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_1_1*
T0*
data_formatNHWC
Њ
@DeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/group_deps_1NoOp<^DeepHit/gradients/DeepHit/Output/BiasAdd_grad/BiasAddGrad_16^DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_1_1
°
HDeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/control_dependency_2Identity5DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_1_1A^DeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/group_deps_1*
T0*H
_class>
<:loc:@DeepHit/gradients/DeepHit/Output/Softmax_grad/mul_1_1
ѓ
JDeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/control_dependency_1_1Identity;DeepHit/gradients/DeepHit/Output/BiasAdd_grad/BiasAddGrad_1A^DeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/group_deps_1*
T0*N
_classD
B@loc:@DeepHit/gradients/DeepHit/Output/BiasAdd_grad/BiasAddGrad_1
ё
5DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_2MatMulHDeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/control_dependency_2$DeepHit/Output/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b(
”
7DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_1_1MatMulDeepHit/dropout_2/Mul_1HDeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/control_dependency_2*
T0*
transpose_a(*
transpose_b( 
є
?DeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/group_deps_1NoOp8^DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_1_16^DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_2
Я
GDeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/control_dependency_2Identity5DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_2@^DeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/group_deps_1*
T0*H
_class>
<:loc:@DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_2
•
IDeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/control_dependency_1_1Identity7DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_1_1@^DeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/group_deps_1*
T0*J
_class@
><loc:@DeepHit/gradients/DeepHit/Output/MatMul_grad/MatMul_1_1
s
4DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape_2ShapeDeepHit/dropout_2/RealDiv_1*
T0*
out_type0
r
6DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape_1_1ShapeDeepHit/dropout_2/Cast_1*
T0*
out_type0
‘
DDeepHit/gradients/DeepHit/dropout_2/Mul_grad/BroadcastGradientArgs_1BroadcastGradientArgs4DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape_26DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape_1_1*
T0
•
2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Mul_2MulGDeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/control_dependency_2DeepHit/dropout_2/Cast_1*
T0
ў
2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Sum_2Sum2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Mul_2DDeepHit/gradients/DeepHit/dropout_2/Mul_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
¬
6DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_2Reshape2DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Sum_24DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape_2*
T0*
Tshape0
™
4DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Mul_1_1MulDeepHit/dropout_2/RealDiv_1GDeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/control_dependency_2*
T0
я
4DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Sum_1_1Sum4DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Mul_1_1FDeepHit/gradients/DeepHit/dropout_2/Mul_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
»
8DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_1_1Reshape4DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Sum_1_16DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Shape_1_1*
T0*
Tshape0
ї
?DeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/group_deps_1NoOp9^DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_1_17^DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_2
°
GDeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/control_dependency_2Identity6DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_2@^DeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/group_deps_1*
T0*I
_class?
=;loc:@DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_2
І
IDeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/control_dependency_1_1Identity8DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_1_1@^DeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/group_deps_1*
T0*K
_classA
?=loc:@DeepHit/gradients/DeepHit/dropout_2/Mul_grad/Reshape_1_1
n
8DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape_2ShapeDeepHit/Reshape_10*
T0*
out_type0
c
:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape_1_1Const*
dtype0*
valueB 
а
HDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/BroadcastGradientArgs_1BroadcastGradientArgs8DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape_2:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape_1_1*
T0
∞
:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDiv_3RealDivGDeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/control_dependency_2DeepHit/dropout_2/Sub_1*
T0
й
6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Sum_2Sum:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDiv_3HDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
ќ
:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_2Reshape6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Sum_28DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape_2*
T0*
Tshape0
Z
6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Neg_1NegDeepHit/Reshape_10*
T0
°
<DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDiv_1_1RealDiv6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Neg_1DeepHit/dropout_2/Sub_1*
T0
І
<DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDiv_2_1RealDiv<DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDiv_1_1DeepHit/dropout_2/Sub_1*
T0
Ќ
6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/mul_1MulGDeepHit/gradients/DeepHit/dropout_2/Mul_grad/tuple/control_dependency_2<DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/RealDiv_2_1*
T0
й
8DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Sum_1_1Sum6DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/mul_1JDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
‘
<DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_1_1Reshape8DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Sum_1_1:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Shape_1_1*
T0*
Tshape0
«
CDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/group_deps_1NoOp=^DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_1_1;^DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_2
±
KDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/control_dependency_2Identity:DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_2D^DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/group_deps_1*
T0*M
_classC
A?loc:@DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_2
Ј
MDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/control_dependency_1_1Identity<DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_1_1D^DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/group_deps_1*
T0*O
_classE
CAloc:@DeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/Reshape_1_1
Ъ
DeepHit/gradients/AddN_3_1AddNBDeepHit/gradients/DeepHit/Output/kernel/Regularizer/Abs_grad/mul_1IDeepHit/gradients/DeepHit/Output/MatMul_grad/tuple/control_dependency_1_1*
N*
T0*U
_classK
IGloc:@DeepHit/gradients/DeepHit/Output/kernel/Regularizer/Abs_grad/mul_1
a
.DeepHit/gradients/DeepHit/Reshape_grad/Shape_1ShapeDeepHit/stack_3*
T0*
out_type0
ѕ
0DeepHit/gradients/DeepHit/Reshape_grad/Reshape_1ReshapeKDeepHit/gradients/DeepHit/dropout_2/RealDiv_grad/tuple/control_dependency_2.DeepHit/gradients/DeepHit/Reshape_grad/Shape_1*
T0*
Tshape0
К
.DeepHit/gradients/DeepHit/stack_grad/unstack_1Unpack0DeepHit/gradients/DeepHit/Reshape_grad/Reshape_1*
T0*

axis*	
num
p
7DeepHit/gradients/DeepHit/stack_grad/tuple/group_deps_1NoOp/^DeepHit/gradients/DeepHit/stack_grad/unstack_1
Б
?DeepHit/gradients/DeepHit/stack_grad/tuple/control_dependency_2Identity.DeepHit/gradients/DeepHit/stack_grad/unstack_18^DeepHit/gradients/DeepHit/stack_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/stack_grad/unstack_1
Е
ADeepHit/gradients/DeepHit/stack_grad/tuple/control_dependency_1_1Identity0DeepHit/gradients/DeepHit/stack_grad/unstack_1:18^DeepHit/gradients/DeepHit/stack_grad/tuple/group_deps_1*
T0*A
_class7
53loc:@DeepHit/gradients/DeepHit/stack_grad/unstack_1
і
>DeepHit/gradients/DeepHit/fully_connected_3/Elu_grad/EluGrad_1EluGrad?DeepHit/gradients/DeepHit/stack_grad/tuple/control_dependency_2DeepHit/fully_connected_3/Elu_1*
T0
ґ
>DeepHit/gradients/DeepHit/fully_connected_4/Elu_grad/EluGrad_1EluGradADeepHit/gradients/DeepHit/stack_grad/tuple/control_dependency_1_1DeepHit/fully_connected_4/Elu_1*
T0
µ
FDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/BiasAddGrad_1BiasAddGrad>DeepHit/gradients/DeepHit/fully_connected_3/Elu_grad/EluGrad_1*
T0*
data_formatNHWC
Ё
KDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/group_deps_1NoOpG^DeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/BiasAddGrad_1?^DeepHit/gradients/DeepHit/fully_connected_3/Elu_grad/EluGrad_1
…
SDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/control_dependency_2Identity>DeepHit/gradients/DeepHit/fully_connected_3/Elu_grad/EluGrad_1L^DeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/group_deps_1*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected_3/Elu_grad/EluGrad_1
џ
UDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/control_dependency_1_1IdentityFDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/BiasAddGrad_1L^DeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/group_deps_1*
T0*Y
_classO
MKloc:@DeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/BiasAddGrad_1
µ
FDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/BiasAddGrad_1BiasAddGrad>DeepHit/gradients/DeepHit/fully_connected_4/Elu_grad/EluGrad_1*
T0*
data_formatNHWC
Ё
KDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/group_deps_1NoOpG^DeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/BiasAddGrad_1?^DeepHit/gradients/DeepHit/fully_connected_4/Elu_grad/EluGrad_1
…
SDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/control_dependency_2Identity>DeepHit/gradients/DeepHit/fully_connected_4/Elu_grad/EluGrad_1L^DeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/group_deps_1*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected_4/Elu_grad/EluGrad_1
џ
UDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/control_dependency_1_1IdentityFDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/BiasAddGrad_1L^DeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/group_deps_1*
T0*Y
_classO
MKloc:@DeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/BiasAddGrad_1
€
@DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_2MatMulSDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/control_dependency_2/DeepHit/fully_connected_3/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b(
в
BDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_1_1MatMulDeepHit/concat_1SDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/control_dependency_2*
T0*
transpose_a(*
transpose_b( 
Џ
JDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/group_deps_1NoOpC^DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_1_1A^DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_2
Ћ
RDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/control_dependency_2Identity@DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_2K^DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/group_deps_1*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_2
—
TDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/control_dependency_1_1IdentityBDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_1_1K^DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/group_deps_1*
T0*U
_classK
IGloc:@DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_1_1
€
@DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_2MatMulSDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/control_dependency_2/DeepHit/fully_connected_4/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b(
в
BDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_1_1MatMulDeepHit/concat_1SDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/control_dependency_2*
T0*
transpose_a(*
transpose_b( 
Џ
JDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/group_deps_1NoOpC^DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_1_1A^DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_2
Ћ
RDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/control_dependency_2Identity@DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_2K^DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/group_deps_1*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_2
—
TDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/control_dependency_1_1IdentityBDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_1_1K^DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/group_deps_1*
T0*U
_classK
IGloc:@DeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/MatMul_1_1
±
DeepHit/gradients/AddN_4_1AddNRDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/control_dependency_2RDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/control_dependency_2*
N*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/MatMul_2
V
,DeepHit/gradients/DeepHit/concat_grad/Rank_1Const*
dtype0*
value	B :
Е
+DeepHit/gradients/DeepHit/concat_grad/mod_1FloorModDeepHit/concat/axis_1,DeepHit/gradients/DeepHit/concat_grad/Rank_1*
T0
a
-DeepHit/gradients/DeepHit/concat_grad/Shape_1ShapeDeepHit/inputs_1*
T0*
out_type0
Н
.DeepHit/gradients/DeepHit/concat_grad/ShapeN_1ShapeNDeepHit/inputs_1DeepHit/fully_connected_2/Elu_1*
N*
T0*
out_type0
№
4DeepHit/gradients/DeepHit/concat_grad/ConcatOffset_1ConcatOffset+DeepHit/gradients/DeepHit/concat_grad/mod_1.DeepHit/gradients/DeepHit/concat_grad/ShapeN_10DeepHit/gradients/DeepHit/concat_grad/ShapeN_1:1*
N
ќ
-DeepHit/gradients/DeepHit/concat_grad/Slice_2SliceDeepHit/gradients/AddN_4_14DeepHit/gradients/DeepHit/concat_grad/ConcatOffset_1.DeepHit/gradients/DeepHit/concat_grad/ShapeN_1*
Index0*
T0
‘
/DeepHit/gradients/DeepHit/concat_grad/Slice_1_1SliceDeepHit/gradients/AddN_4_16DeepHit/gradients/DeepHit/concat_grad/ConcatOffset_1:10DeepHit/gradients/DeepHit/concat_grad/ShapeN_1:1*
Index0*
T0
Ґ
8DeepHit/gradients/DeepHit/concat_grad/tuple/group_deps_1NoOp0^DeepHit/gradients/DeepHit/concat_grad/Slice_1_1.^DeepHit/gradients/DeepHit/concat_grad/Slice_2
Б
@DeepHit/gradients/DeepHit/concat_grad/tuple/control_dependency_2Identity-DeepHit/gradients/DeepHit/concat_grad/Slice_29^DeepHit/gradients/DeepHit/concat_grad/tuple/group_deps_1*
T0*@
_class6
42loc:@DeepHit/gradients/DeepHit/concat_grad/Slice_2
З
BDeepHit/gradients/DeepHit/concat_grad/tuple/control_dependency_1_1Identity/DeepHit/gradients/DeepHit/concat_grad/Slice_1_19^DeepHit/gradients/DeepHit/concat_grad/tuple/group_deps_1*
T0*B
_class8
64loc:@DeepHit/gradients/DeepHit/concat_grad/Slice_1_1
≈
DeepHit/gradients/AddN_5_1AddNRDeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Mul_1_1TDeepHit/gradients/DeepHit/fully_connected_3/MatMul_grad/tuple/control_dependency_1_1*
N*
T0*e
_class[
YWloc:@DeepHit/gradients/DeepHit/fully_connected_3/kernel/Regularizer/Square_grad/Mul_1_1
Ј
>DeepHit/gradients/DeepHit/fully_connected_2/Elu_grad/EluGrad_1EluGradBDeepHit/gradients/DeepHit/concat_grad/tuple/control_dependency_1_1DeepHit/fully_connected_2/Elu_1*
T0
≈
DeepHit/gradients/AddN_6_1AddNRDeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Mul_1_1TDeepHit/gradients/DeepHit/fully_connected_4/MatMul_grad/tuple/control_dependency_1_1*
N*
T0*e
_class[
YWloc:@DeepHit/gradients/DeepHit/fully_connected_4/kernel/Regularizer/Square_grad/Mul_1_1
µ
FDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/BiasAddGrad_1BiasAddGrad>DeepHit/gradients/DeepHit/fully_connected_2/Elu_grad/EluGrad_1*
T0*
data_formatNHWC
Ё
KDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/group_deps_1NoOpG^DeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/BiasAddGrad_1?^DeepHit/gradients/DeepHit/fully_connected_2/Elu_grad/EluGrad_1
…
SDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/control_dependency_2Identity>DeepHit/gradients/DeepHit/fully_connected_2/Elu_grad/EluGrad_1L^DeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/group_deps_1*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected_2/Elu_grad/EluGrad_1
џ
UDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1_1IdentityFDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/BiasAddGrad_1L^DeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/group_deps_1*
T0*Y
_classO
MKloc:@DeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/BiasAddGrad_1
€
@DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_2MatMulSDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/control_dependency_2/DeepHit/fully_connected_2/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b(
й
BDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_1_1MatMulDeepHit/dropout_1/Mul_1SDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/control_dependency_2*
T0*
transpose_a(*
transpose_b( 
Џ
JDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/group_deps_1NoOpC^DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_1_1A^DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_2
Ћ
RDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/control_dependency_2Identity@DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_2K^DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/group_deps_1*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_2
—
TDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/control_dependency_1_1IdentityBDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_1_1K^DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/group_deps_1*
T0*U
_classK
IGloc:@DeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/MatMul_1_1
s
4DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape_2ShapeDeepHit/dropout_1/RealDiv_1*
T0*
out_type0
r
6DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape_1_1ShapeDeepHit/dropout_1/Cast_1*
T0*
out_type0
‘
DDeepHit/gradients/DeepHit/dropout_1/Mul_grad/BroadcastGradientArgs_1BroadcastGradientArgs4DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape_26DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape_1_1*
T0
∞
2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Mul_2MulRDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/control_dependency_2DeepHit/dropout_1/Cast_1*
T0
ў
2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Sum_2Sum2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Mul_2DDeepHit/gradients/DeepHit/dropout_1/Mul_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
¬
6DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_2Reshape2DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Sum_24DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape_2*
T0*
Tshape0
µ
4DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Mul_1_1MulDeepHit/dropout_1/RealDiv_1RDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/control_dependency_2*
T0
я
4DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Sum_1_1Sum4DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Mul_1_1FDeepHit/gradients/DeepHit/dropout_1/Mul_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
»
8DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_1_1Reshape4DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Sum_1_16DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Shape_1_1*
T0*
Tshape0
ї
?DeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/group_deps_1NoOp9^DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_1_17^DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_2
°
GDeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/control_dependency_2Identity6DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_2@^DeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/group_deps_1*
T0*I
_class?
=;loc:@DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_2
І
IDeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/control_dependency_1_1Identity8DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_1_1@^DeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/group_deps_1*
T0*K
_classA
?=loc:@DeepHit/gradients/DeepHit/dropout_1/Mul_grad/Reshape_1_1
{
8DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape_2ShapeDeepHit/fully_connected_1/Elu_1*
T0*
out_type0
c
:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape_1_1Const*
dtype0*
valueB 
а
HDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/BroadcastGradientArgs_1BroadcastGradientArgs8DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape_2:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape_1_1*
T0
∞
:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDiv_3RealDivGDeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/control_dependency_2DeepHit/dropout_1/Sub_1*
T0
й
6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Sum_2Sum:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDiv_3HDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
ќ
:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_2Reshape6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Sum_28DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape_2*
T0*
Tshape0
g
6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Neg_1NegDeepHit/fully_connected_1/Elu_1*
T0
°
<DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDiv_1_1RealDiv6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Neg_1DeepHit/dropout_1/Sub_1*
T0
І
<DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDiv_2_1RealDiv<DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDiv_1_1DeepHit/dropout_1/Sub_1*
T0
Ќ
6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/mul_1MulGDeepHit/gradients/DeepHit/dropout_1/Mul_grad/tuple/control_dependency_2<DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/RealDiv_2_1*
T0
й
8DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Sum_1_1Sum6DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/mul_1JDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
‘
<DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_1_1Reshape8DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Sum_1_1:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Shape_1_1*
T0*
Tshape0
«
CDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/group_deps_1NoOp=^DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_1_1;^DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_2
±
KDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/control_dependency_2Identity:DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_2D^DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/group_deps_1*
T0*M
_classC
A?loc:@DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_2
Ј
MDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/control_dependency_1_1Identity<DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_1_1D^DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/group_deps_1*
T0*O
_classE
CAloc:@DeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/Reshape_1_1
≈
DeepHit/gradients/AddN_7_1AddNRDeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Mul_1_1TDeepHit/gradients/DeepHit/fully_connected_2/MatMul_grad/tuple/control_dependency_1_1*
N*
T0*e
_class[
YWloc:@DeepHit/gradients/DeepHit/fully_connected_2/kernel/Regularizer/Square_grad/Mul_1_1
ј
>DeepHit/gradients/DeepHit/fully_connected_1/Elu_grad/EluGrad_1EluGradKDeepHit/gradients/DeepHit/dropout_1/RealDiv_grad/tuple/control_dependency_2DeepHit/fully_connected_1/Elu_1*
T0
µ
FDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/BiasAddGrad_1BiasAddGrad>DeepHit/gradients/DeepHit/fully_connected_1/Elu_grad/EluGrad_1*
T0*
data_formatNHWC
Ё
KDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/group_deps_1NoOpG^DeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/BiasAddGrad_1?^DeepHit/gradients/DeepHit/fully_connected_1/Elu_grad/EluGrad_1
…
SDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/control_dependency_2Identity>DeepHit/gradients/DeepHit/fully_connected_1/Elu_grad/EluGrad_1L^DeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/group_deps_1*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected_1/Elu_grad/EluGrad_1
џ
UDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1_1IdentityFDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/BiasAddGrad_1L^DeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/group_deps_1*
T0*Y
_classO
MKloc:@DeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/BiasAddGrad_1
€
@DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_2MatMulSDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/control_dependency_2/DeepHit/fully_connected_1/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b(
з
BDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_1_1MatMulDeepHit/dropout/Mul_1SDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/control_dependency_2*
T0*
transpose_a(*
transpose_b( 
Џ
JDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/group_deps_1NoOpC^DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_1_1A^DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_2
Ћ
RDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/control_dependency_2Identity@DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_2K^DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/group_deps_1*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_2
—
TDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/control_dependency_1_1IdentityBDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_1_1K^DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/group_deps_1*
T0*U
_classK
IGloc:@DeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/MatMul_1_1
o
2DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape_2ShapeDeepHit/dropout/RealDiv_1*
T0*
out_type0
n
4DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape_1_1ShapeDeepHit/dropout/Cast_1*
T0*
out_type0
ќ
BDeepHit/gradients/DeepHit/dropout/Mul_grad/BroadcastGradientArgs_1BroadcastGradientArgs2DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape_24DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape_1_1*
T0
ђ
0DeepHit/gradients/DeepHit/dropout/Mul_grad/Mul_2MulRDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/control_dependency_2DeepHit/dropout/Cast_1*
T0
”
0DeepHit/gradients/DeepHit/dropout/Mul_grad/Sum_2Sum0DeepHit/gradients/DeepHit/dropout/Mul_grad/Mul_2BDeepHit/gradients/DeepHit/dropout/Mul_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
Љ
4DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_2Reshape0DeepHit/gradients/DeepHit/dropout/Mul_grad/Sum_22DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape_2*
T0*
Tshape0
±
2DeepHit/gradients/DeepHit/dropout/Mul_grad/Mul_1_1MulDeepHit/dropout/RealDiv_1RDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/control_dependency_2*
T0
ў
2DeepHit/gradients/DeepHit/dropout/Mul_grad/Sum_1_1Sum2DeepHit/gradients/DeepHit/dropout/Mul_grad/Mul_1_1DDeepHit/gradients/DeepHit/dropout/Mul_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
¬
6DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_1_1Reshape2DeepHit/gradients/DeepHit/dropout/Mul_grad/Sum_1_14DeepHit/gradients/DeepHit/dropout/Mul_grad/Shape_1_1*
T0*
Tshape0
µ
=DeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/group_deps_1NoOp7^DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_1_15^DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_2
Щ
EDeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/control_dependency_2Identity4DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_2>^DeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/group_deps_1*
T0*G
_class=
;9loc:@DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_2
Я
GDeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/control_dependency_1_1Identity6DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_1_1>^DeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/group_deps_1*
T0*I
_class?
=;loc:@DeepHit/gradients/DeepHit/dropout/Mul_grad/Reshape_1_1
w
6DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape_2ShapeDeepHit/fully_connected/Elu_1*
T0*
out_type0
a
8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape_1_1Const*
dtype0*
valueB 
Џ
FDeepHit/gradients/DeepHit/dropout/RealDiv_grad/BroadcastGradientArgs_1BroadcastGradientArgs6DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape_28DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape_1_1*
T0
™
8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDiv_3RealDivEDeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/control_dependency_2DeepHit/dropout/Sub_1*
T0
г
4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Sum_2Sum8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDiv_3FDeepHit/gradients/DeepHit/dropout/RealDiv_grad/BroadcastGradientArgs_1*
T0*

Tidx0*
	keep_dims( 
»
8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_2Reshape4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Sum_26DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape_2*
T0*
Tshape0
c
4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Neg_1NegDeepHit/fully_connected/Elu_1*
T0
Ы
:DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDiv_1_1RealDiv4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Neg_1DeepHit/dropout/Sub_1*
T0
°
:DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDiv_2_1RealDiv:DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDiv_1_1DeepHit/dropout/Sub_1*
T0
«
4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/mul_1MulEDeepHit/gradients/DeepHit/dropout/Mul_grad/tuple/control_dependency_2:DeepHit/gradients/DeepHit/dropout/RealDiv_grad/RealDiv_2_1*
T0
г
6DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Sum_1_1Sum4DeepHit/gradients/DeepHit/dropout/RealDiv_grad/mul_1HDeepHit/gradients/DeepHit/dropout/RealDiv_grad/BroadcastGradientArgs_1:1*
T0*

Tidx0*
	keep_dims( 
ќ
:DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_1_1Reshape6DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Sum_1_18DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Shape_1_1*
T0*
Tshape0
Ѕ
ADeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/group_deps_1NoOp;^DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_1_19^DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_2
©
IDeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/control_dependency_2Identity8DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_2B^DeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/group_deps_1*
T0*K
_classA
?=loc:@DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_2
ѓ
KDeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/control_dependency_1_1Identity:DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_1_1B^DeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/group_deps_1*
T0*M
_classC
A?loc:@DeepHit/gradients/DeepHit/dropout/RealDiv_grad/Reshape_1_1
≈
DeepHit/gradients/AddN_8_1AddNRDeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Mul_1_1TDeepHit/gradients/DeepHit/fully_connected_1/MatMul_grad/tuple/control_dependency_1_1*
N*
T0*e
_class[
YWloc:@DeepHit/gradients/DeepHit/fully_connected_1/kernel/Regularizer/Square_grad/Mul_1_1
Ї
<DeepHit/gradients/DeepHit/fully_connected/Elu_grad/EluGrad_1EluGradIDeepHit/gradients/DeepHit/dropout/RealDiv_grad/tuple/control_dependency_2DeepHit/fully_connected/Elu_1*
T0
±
DDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/BiasAddGrad_1BiasAddGrad<DeepHit/gradients/DeepHit/fully_connected/Elu_grad/EluGrad_1*
T0*
data_formatNHWC
„
IDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/group_deps_1NoOpE^DeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/BiasAddGrad_1=^DeepHit/gradients/DeepHit/fully_connected/Elu_grad/EluGrad_1
Ѕ
QDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/control_dependency_2Identity<DeepHit/gradients/DeepHit/fully_connected/Elu_grad/EluGrad_1J^DeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/group_deps_1*
T0*O
_classE
CAloc:@DeepHit/gradients/DeepHit/fully_connected/Elu_grad/EluGrad_1
”
SDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/control_dependency_1_1IdentityDDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/BiasAddGrad_1J^DeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/group_deps_1*
T0*W
_classM
KIloc:@DeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/BiasAddGrad_1
щ
>DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_2MatMulQDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/control_dependency_2-DeepHit/fully_connected/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b(
ё
@DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_1_1MatMulDeepHit/inputs_1QDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/control_dependency_2*
T0*
transpose_a(*
transpose_b( 
‘
HDeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/group_deps_1NoOpA^DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_1_1?^DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_2
√
PDeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/control_dependency_2Identity>DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_2I^DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/group_deps_1*
T0*Q
_classG
ECloc:@DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_2
…
RDeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/control_dependency_1_1Identity@DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_1_1I^DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/group_deps_1*
T0*S
_classI
GEloc:@DeepHit/gradients/DeepHit/fully_connected/MatMul_grad/MatMul_1_1
њ
DeepHit/gradients/AddN_9_1AddNPDeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Mul_1_1RDeepHit/gradients/DeepHit/fully_connected/MatMul_grad/tuple/control_dependency_1_1*
N*
T0*c
_classY
WUloc:@DeepHit/gradients/DeepHit/fully_connected/kernel/Regularizer/Square_grad/Mul_1_1
Ж
-DeepHit/beta1_power/Initializer/initial_valueConst**
_class 
loc:@DeepHit/Output/biases_1*
dtype0*
valueB
 *fff?
Є
DeepHit/beta1_power_1VarHandleOp**
_class 
loc:@DeepHit/Output/biases_1*
allowed_devices
 *
	container *
dtype0*
shape: *$
shared_nameDeepHit/beta1_power
Н
4DeepHit/beta1_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpDeepHit/beta1_power_1**
_class 
loc:@DeepHit/Output/biases_1
Щ
DeepHit/beta1_power/Assign_1AssignVariableOpDeepHit/beta1_power_1-DeepHit/beta1_power/Initializer/initial_value*
dtype0*
validate_shape( 
Й
'DeepHit/beta1_power/Read/ReadVariableOpReadVariableOpDeepHit/beta1_power_1**
_class 
loc:@DeepHit/Output/biases_1*
dtype0
Ж
-DeepHit/beta2_power/Initializer/initial_valueConst**
_class 
loc:@DeepHit/Output/biases_1*
dtype0*
valueB
 *wЊ?
Є
DeepHit/beta2_power_1VarHandleOp**
_class 
loc:@DeepHit/Output/biases_1*
allowed_devices
 *
	container *
dtype0*
shape: *$
shared_nameDeepHit/beta2_power
Н
4DeepHit/beta2_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpDeepHit/beta2_power_1**
_class 
loc:@DeepHit/Output/biases_1
Щ
DeepHit/beta2_power/Assign_1AssignVariableOpDeepHit/beta2_power_1-DeepHit/beta2_power/Initializer/initial_value*
dtype0*
validate_shape( 
Й
'DeepHit/beta2_power/Read/ReadVariableOpReadVariableOpDeepHit/beta2_power_1**
_class 
loc:@DeepHit/Output/biases_1*
dtype0
Ђ
@DeepHit/DeepHit/fully_connected/weights/Adam/Initializer/zeros_1Const*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
dtype0*
valueB`
*    
ь
.DeepHit/DeepHit/fully_connected/weights/Adam_2VarHandleOp*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:`
*=
shared_name.,DeepHit/DeepHit/fully_connected/weights/Adam
…
MDeepHit/DeepHit/fully_connected/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp.DeepHit/DeepHit/fully_connected/weights/Adam_2*4
_class*
(&loc:@DeepHit/fully_connected/weights_1
ё
5DeepHit/DeepHit/fully_connected/weights/Adam/Assign_1AssignVariableOp.DeepHit/DeepHit/fully_connected/weights/Adam_2@DeepHit/DeepHit/fully_connected/weights/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
≈
@DeepHit/DeepHit/fully_connected/weights/Adam/Read/ReadVariableOpReadVariableOp.DeepHit/DeepHit/fully_connected/weights/Adam_2*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
dtype0
≠
BDeepHit/DeepHit/fully_connected/weights/Adam_1/Initializer/zeros_1Const*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
dtype0*
valueB`
*    
А
0DeepHit/DeepHit/fully_connected/weights/Adam_1_1VarHandleOp*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:`
*?
shared_name0.DeepHit/DeepHit/fully_connected/weights/Adam_1
Ќ
ODeepHit/DeepHit/fully_connected/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp0DeepHit/DeepHit/fully_connected/weights/Adam_1_1*4
_class*
(&loc:@DeepHit/fully_connected/weights_1
д
7DeepHit/DeepHit/fully_connected/weights/Adam_1/Assign_1AssignVariableOp0DeepHit/DeepHit/fully_connected/weights/Adam_1_1BDeepHit/DeepHit/fully_connected/weights/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
…
BDeepHit/DeepHit/fully_connected/weights/Adam_1/Read/ReadVariableOpReadVariableOp0DeepHit/DeepHit/fully_connected/weights/Adam_1_1*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
dtype0
•
?DeepHit/DeepHit/fully_connected/biases/Adam/Initializer/zeros_1Const*3
_class)
'%loc:@DeepHit/fully_connected/biases_1*
dtype0*
valueB
*    
х
-DeepHit/DeepHit/fully_connected/biases/Adam_2VarHandleOp*3
_class)
'%loc:@DeepHit/fully_connected/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:
*<
shared_name-+DeepHit/DeepHit/fully_connected/biases/Adam
∆
LDeepHit/DeepHit/fully_connected/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp-DeepHit/DeepHit/fully_connected/biases/Adam_2*3
_class)
'%loc:@DeepHit/fully_connected/biases_1
џ
4DeepHit/DeepHit/fully_connected/biases/Adam/Assign_1AssignVariableOp-DeepHit/DeepHit/fully_connected/biases/Adam_2?DeepHit/DeepHit/fully_connected/biases/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
¬
?DeepHit/DeepHit/fully_connected/biases/Adam/Read/ReadVariableOpReadVariableOp-DeepHit/DeepHit/fully_connected/biases/Adam_2*3
_class)
'%loc:@DeepHit/fully_connected/biases_1*
dtype0
І
ADeepHit/DeepHit/fully_connected/biases/Adam_1/Initializer/zeros_1Const*3
_class)
'%loc:@DeepHit/fully_connected/biases_1*
dtype0*
valueB
*    
щ
/DeepHit/DeepHit/fully_connected/biases/Adam_1_1VarHandleOp*3
_class)
'%loc:@DeepHit/fully_connected/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:
*>
shared_name/-DeepHit/DeepHit/fully_connected/biases/Adam_1
 
NDeepHit/DeepHit/fully_connected/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp/DeepHit/DeepHit/fully_connected/biases/Adam_1_1*3
_class)
'%loc:@DeepHit/fully_connected/biases_1
б
6DeepHit/DeepHit/fully_connected/biases/Adam_1/Assign_1AssignVariableOp/DeepHit/DeepHit/fully_connected/biases/Adam_1_1ADeepHit/DeepHit/fully_connected/biases/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
∆
ADeepHit/DeepHit/fully_connected/biases/Adam_1/Read/ReadVariableOpReadVariableOp/DeepHit/DeepHit/fully_connected/biases/Adam_1_1*3
_class)
'%loc:@DeepHit/fully_connected/biases_1*
dtype0
ѓ
BDeepHit/DeepHit/fully_connected_1/weights/Adam/Initializer/zeros_1Const*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
dtype0*
valueB

*    
В
0DeepHit/DeepHit/fully_connected_1/weights/Adam_2VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:

*?
shared_name0.DeepHit/DeepHit/fully_connected_1/weights/Adam
ѕ
ODeepHit/DeepHit/fully_connected_1/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp0DeepHit/DeepHit/fully_connected_1/weights/Adam_2*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1
д
7DeepHit/DeepHit/fully_connected_1/weights/Adam/Assign_1AssignVariableOp0DeepHit/DeepHit/fully_connected_1/weights/Adam_2BDeepHit/DeepHit/fully_connected_1/weights/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
Ћ
BDeepHit/DeepHit/fully_connected_1/weights/Adam/Read/ReadVariableOpReadVariableOp0DeepHit/DeepHit/fully_connected_1/weights/Adam_2*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
dtype0
±
DDeepHit/DeepHit/fully_connected_1/weights/Adam_1/Initializer/zeros_1Const*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
dtype0*
valueB

*    
Ж
2DeepHit/DeepHit/fully_connected_1/weights/Adam_1_1VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:

*A
shared_name20DeepHit/DeepHit/fully_connected_1/weights/Adam_1
”
QDeepHit/DeepHit/fully_connected_1/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp2DeepHit/DeepHit/fully_connected_1/weights/Adam_1_1*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1
к
9DeepHit/DeepHit/fully_connected_1/weights/Adam_1/Assign_1AssignVariableOp2DeepHit/DeepHit/fully_connected_1/weights/Adam_1_1DDeepHit/DeepHit/fully_connected_1/weights/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
ѕ
DDeepHit/DeepHit/fully_connected_1/weights/Adam_1/Read/ReadVariableOpReadVariableOp2DeepHit/DeepHit/fully_connected_1/weights/Adam_1_1*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
dtype0
©
ADeepHit/DeepHit/fully_connected_1/biases/Adam/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1*
dtype0*
valueB
*    
ы
/DeepHit/DeepHit/fully_connected_1/biases/Adam_2VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:
*>
shared_name/-DeepHit/DeepHit/fully_connected_1/biases/Adam
ћ
NDeepHit/DeepHit/fully_connected_1/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp/DeepHit/DeepHit/fully_connected_1/biases/Adam_2*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1
б
6DeepHit/DeepHit/fully_connected_1/biases/Adam/Assign_1AssignVariableOp/DeepHit/DeepHit/fully_connected_1/biases/Adam_2ADeepHit/DeepHit/fully_connected_1/biases/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
»
ADeepHit/DeepHit/fully_connected_1/biases/Adam/Read/ReadVariableOpReadVariableOp/DeepHit/DeepHit/fully_connected_1/biases/Adam_2*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1*
dtype0
Ђ
CDeepHit/DeepHit/fully_connected_1/biases/Adam_1/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1*
dtype0*
valueB
*    
€
1DeepHit/DeepHit/fully_connected_1/biases/Adam_1_1VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:
*@
shared_name1/DeepHit/DeepHit/fully_connected_1/biases/Adam_1
–
PDeepHit/DeepHit/fully_connected_1/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp1DeepHit/DeepHit/fully_connected_1/biases/Adam_1_1*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1
з
8DeepHit/DeepHit/fully_connected_1/biases/Adam_1/Assign_1AssignVariableOp1DeepHit/DeepHit/fully_connected_1/biases/Adam_1_1CDeepHit/DeepHit/fully_connected_1/biases/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
ћ
CDeepHit/DeepHit/fully_connected_1/biases/Adam_1/Read/ReadVariableOpReadVariableOp1DeepHit/DeepHit/fully_connected_1/biases/Adam_1_1*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1*
dtype0
ѓ
BDeepHit/DeepHit/fully_connected_2/weights/Adam/Initializer/zeros_1Const*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
dtype0*
valueB

*    
В
0DeepHit/DeepHit/fully_connected_2/weights/Adam_2VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:

*?
shared_name0.DeepHit/DeepHit/fully_connected_2/weights/Adam
ѕ
ODeepHit/DeepHit/fully_connected_2/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp0DeepHit/DeepHit/fully_connected_2/weights/Adam_2*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1
д
7DeepHit/DeepHit/fully_connected_2/weights/Adam/Assign_1AssignVariableOp0DeepHit/DeepHit/fully_connected_2/weights/Adam_2BDeepHit/DeepHit/fully_connected_2/weights/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
Ћ
BDeepHit/DeepHit/fully_connected_2/weights/Adam/Read/ReadVariableOpReadVariableOp0DeepHit/DeepHit/fully_connected_2/weights/Adam_2*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
dtype0
±
DDeepHit/DeepHit/fully_connected_2/weights/Adam_1/Initializer/zeros_1Const*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
dtype0*
valueB

*    
Ж
2DeepHit/DeepHit/fully_connected_2/weights/Adam_1_1VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:

*A
shared_name20DeepHit/DeepHit/fully_connected_2/weights/Adam_1
”
QDeepHit/DeepHit/fully_connected_2/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp2DeepHit/DeepHit/fully_connected_2/weights/Adam_1_1*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1
к
9DeepHit/DeepHit/fully_connected_2/weights/Adam_1/Assign_1AssignVariableOp2DeepHit/DeepHit/fully_connected_2/weights/Adam_1_1DDeepHit/DeepHit/fully_connected_2/weights/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
ѕ
DDeepHit/DeepHit/fully_connected_2/weights/Adam_1/Read/ReadVariableOpReadVariableOp2DeepHit/DeepHit/fully_connected_2/weights/Adam_1_1*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
dtype0
©
ADeepHit/DeepHit/fully_connected_2/biases/Adam/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1*
dtype0*
valueB
*    
ы
/DeepHit/DeepHit/fully_connected_2/biases/Adam_2VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:
*>
shared_name/-DeepHit/DeepHit/fully_connected_2/biases/Adam
ћ
NDeepHit/DeepHit/fully_connected_2/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp/DeepHit/DeepHit/fully_connected_2/biases/Adam_2*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1
б
6DeepHit/DeepHit/fully_connected_2/biases/Adam/Assign_1AssignVariableOp/DeepHit/DeepHit/fully_connected_2/biases/Adam_2ADeepHit/DeepHit/fully_connected_2/biases/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
»
ADeepHit/DeepHit/fully_connected_2/biases/Adam/Read/ReadVariableOpReadVariableOp/DeepHit/DeepHit/fully_connected_2/biases/Adam_2*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1*
dtype0
Ђ
CDeepHit/DeepHit/fully_connected_2/biases/Adam_1/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1*
dtype0*
valueB
*    
€
1DeepHit/DeepHit/fully_connected_2/biases/Adam_1_1VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:
*@
shared_name1/DeepHit/DeepHit/fully_connected_2/biases/Adam_1
–
PDeepHit/DeepHit/fully_connected_2/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp1DeepHit/DeepHit/fully_connected_2/biases/Adam_1_1*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1
з
8DeepHit/DeepHit/fully_connected_2/biases/Adam_1/Assign_1AssignVariableOp1DeepHit/DeepHit/fully_connected_2/biases/Adam_1_1CDeepHit/DeepHit/fully_connected_2/biases/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
ћ
CDeepHit/DeepHit/fully_connected_2/biases/Adam_1/Read/ReadVariableOpReadVariableOp1DeepHit/DeepHit/fully_connected_2/biases/Adam_1_1*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1*
dtype0
њ
RDeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros/shape_as_tensor_1Const*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
dtype0*
valueB"j      
≠
HDeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros/Const_1Const*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
dtype0*
valueB
 *    
ї
BDeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros_1FillRDeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros/shape_as_tensor_1HDeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros/Const_1*
T0*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*

index_type0
В
0DeepHit/DeepHit/fully_connected_3/weights/Adam_2VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:j*?
shared_name0.DeepHit/DeepHit/fully_connected_3/weights/Adam
ѕ
ODeepHit/DeepHit/fully_connected_3/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp0DeepHit/DeepHit/fully_connected_3/weights/Adam_2*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1
д
7DeepHit/DeepHit/fully_connected_3/weights/Adam/Assign_1AssignVariableOp0DeepHit/DeepHit/fully_connected_3/weights/Adam_2BDeepHit/DeepHit/fully_connected_3/weights/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
Ћ
BDeepHit/DeepHit/fully_connected_3/weights/Adam/Read/ReadVariableOpReadVariableOp0DeepHit/DeepHit/fully_connected_3/weights/Adam_2*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
dtype0
Ѕ
TDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros/shape_as_tensor_1Const*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
dtype0*
valueB"j      
ѓ
JDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros/Const_1Const*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
dtype0*
valueB
 *    
Ѕ
DDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros_1FillTDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros/shape_as_tensor_1JDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros/Const_1*
T0*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*

index_type0
Ж
2DeepHit/DeepHit/fully_connected_3/weights/Adam_1_1VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:j*A
shared_name20DeepHit/DeepHit/fully_connected_3/weights/Adam_1
”
QDeepHit/DeepHit/fully_connected_3/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp2DeepHit/DeepHit/fully_connected_3/weights/Adam_1_1*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1
к
9DeepHit/DeepHit/fully_connected_3/weights/Adam_1/Assign_1AssignVariableOp2DeepHit/DeepHit/fully_connected_3/weights/Adam_1_1DDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
ѕ
DDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Read/ReadVariableOpReadVariableOp2DeepHit/DeepHit/fully_connected_3/weights/Adam_1_1*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
dtype0
©
ADeepHit/DeepHit/fully_connected_3/biases/Adam/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1*
dtype0*
valueB*    
ы
/DeepHit/DeepHit/fully_connected_3/biases/Adam_2VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:*>
shared_name/-DeepHit/DeepHit/fully_connected_3/biases/Adam
ћ
NDeepHit/DeepHit/fully_connected_3/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp/DeepHit/DeepHit/fully_connected_3/biases/Adam_2*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1
б
6DeepHit/DeepHit/fully_connected_3/biases/Adam/Assign_1AssignVariableOp/DeepHit/DeepHit/fully_connected_3/biases/Adam_2ADeepHit/DeepHit/fully_connected_3/biases/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
»
ADeepHit/DeepHit/fully_connected_3/biases/Adam/Read/ReadVariableOpReadVariableOp/DeepHit/DeepHit/fully_connected_3/biases/Adam_2*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1*
dtype0
Ђ
CDeepHit/DeepHit/fully_connected_3/biases/Adam_1/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1*
dtype0*
valueB*    
€
1DeepHit/DeepHit/fully_connected_3/biases/Adam_1_1VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:*@
shared_name1/DeepHit/DeepHit/fully_connected_3/biases/Adam_1
–
PDeepHit/DeepHit/fully_connected_3/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp1DeepHit/DeepHit/fully_connected_3/biases/Adam_1_1*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1
з
8DeepHit/DeepHit/fully_connected_3/biases/Adam_1/Assign_1AssignVariableOp1DeepHit/DeepHit/fully_connected_3/biases/Adam_1_1CDeepHit/DeepHit/fully_connected_3/biases/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
ћ
CDeepHit/DeepHit/fully_connected_3/biases/Adam_1/Read/ReadVariableOpReadVariableOp1DeepHit/DeepHit/fully_connected_3/biases/Adam_1_1*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1*
dtype0
њ
RDeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros/shape_as_tensor_1Const*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
dtype0*
valueB"j      
≠
HDeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros/Const_1Const*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
dtype0*
valueB
 *    
ї
BDeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros_1FillRDeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros/shape_as_tensor_1HDeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros/Const_1*
T0*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*

index_type0
В
0DeepHit/DeepHit/fully_connected_4/weights/Adam_2VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:j*?
shared_name0.DeepHit/DeepHit/fully_connected_4/weights/Adam
ѕ
ODeepHit/DeepHit/fully_connected_4/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp0DeepHit/DeepHit/fully_connected_4/weights/Adam_2*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1
д
7DeepHit/DeepHit/fully_connected_4/weights/Adam/Assign_1AssignVariableOp0DeepHit/DeepHit/fully_connected_4/weights/Adam_2BDeepHit/DeepHit/fully_connected_4/weights/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
Ћ
BDeepHit/DeepHit/fully_connected_4/weights/Adam/Read/ReadVariableOpReadVariableOp0DeepHit/DeepHit/fully_connected_4/weights/Adam_2*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
dtype0
Ѕ
TDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros/shape_as_tensor_1Const*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
dtype0*
valueB"j      
ѓ
JDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros/Const_1Const*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
dtype0*
valueB
 *    
Ѕ
DDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros_1FillTDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros/shape_as_tensor_1JDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros/Const_1*
T0*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*

index_type0
Ж
2DeepHit/DeepHit/fully_connected_4/weights/Adam_1_1VarHandleOp*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
allowed_devices
 *
	container *
dtype0*
shape
:j*A
shared_name20DeepHit/DeepHit/fully_connected_4/weights/Adam_1
”
QDeepHit/DeepHit/fully_connected_4/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp2DeepHit/DeepHit/fully_connected_4/weights/Adam_1_1*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1
к
9DeepHit/DeepHit/fully_connected_4/weights/Adam_1/Assign_1AssignVariableOp2DeepHit/DeepHit/fully_connected_4/weights/Adam_1_1DDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
ѕ
DDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Read/ReadVariableOpReadVariableOp2DeepHit/DeepHit/fully_connected_4/weights/Adam_1_1*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
dtype0
©
ADeepHit/DeepHit/fully_connected_4/biases/Adam/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1*
dtype0*
valueB*    
ы
/DeepHit/DeepHit/fully_connected_4/biases/Adam_2VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:*>
shared_name/-DeepHit/DeepHit/fully_connected_4/biases/Adam
ћ
NDeepHit/DeepHit/fully_connected_4/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp/DeepHit/DeepHit/fully_connected_4/biases/Adam_2*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1
б
6DeepHit/DeepHit/fully_connected_4/biases/Adam/Assign_1AssignVariableOp/DeepHit/DeepHit/fully_connected_4/biases/Adam_2ADeepHit/DeepHit/fully_connected_4/biases/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
»
ADeepHit/DeepHit/fully_connected_4/biases/Adam/Read/ReadVariableOpReadVariableOp/DeepHit/DeepHit/fully_connected_4/biases/Adam_2*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1*
dtype0
Ђ
CDeepHit/DeepHit/fully_connected_4/biases/Adam_1/Initializer/zeros_1Const*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1*
dtype0*
valueB*    
€
1DeepHit/DeepHit/fully_connected_4/biases/Adam_1_1VarHandleOp*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:*@
shared_name1/DeepHit/DeepHit/fully_connected_4/biases/Adam_1
–
PDeepHit/DeepHit/fully_connected_4/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp1DeepHit/DeepHit/fully_connected_4/biases/Adam_1_1*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1
з
8DeepHit/DeepHit/fully_connected_4/biases/Adam_1/Assign_1AssignVariableOp1DeepHit/DeepHit/fully_connected_4/biases/Adam_1_1CDeepHit/DeepHit/fully_connected_4/biases/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
ћ
CDeepHit/DeepHit/fully_connected_4/biases/Adam_1/Read/ReadVariableOpReadVariableOp1DeepHit/DeepHit/fully_connected_4/biases/Adam_1_1*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1*
dtype0
©
GDeepHit/DeepHit/Output/weights/Adam/Initializer/zeros/shape_as_tensor_1Const*+
_class!
loc:@DeepHit/Output/weights_1*
dtype0*
valueB"(     
Ч
=DeepHit/DeepHit/Output/weights/Adam/Initializer/zeros/Const_1Const*+
_class!
loc:@DeepHit/Output/weights_1*
dtype0*
valueB
 *    
П
7DeepHit/DeepHit/Output/weights/Adam/Initializer/zeros_1FillGDeepHit/DeepHit/Output/weights/Adam/Initializer/zeros/shape_as_tensor_1=DeepHit/DeepHit/Output/weights/Adam/Initializer/zeros/Const_1*
T0*+
_class!
loc:@DeepHit/Output/weights_1*

index_type0
в
%DeepHit/DeepHit/Output/weights/Adam_2VarHandleOp*+
_class!
loc:@DeepHit/Output/weights_1*
allowed_devices
 *
	container *
dtype0*
shape:	(Ю*4
shared_name%#DeepHit/DeepHit/Output/weights/Adam
Ѓ
DDeepHit/DeepHit/Output/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp%DeepHit/DeepHit/Output/weights/Adam_2*+
_class!
loc:@DeepHit/Output/weights_1
√
,DeepHit/DeepHit/Output/weights/Adam/Assign_1AssignVariableOp%DeepHit/DeepHit/Output/weights/Adam_27DeepHit/DeepHit/Output/weights/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
™
7DeepHit/DeepHit/Output/weights/Adam/Read/ReadVariableOpReadVariableOp%DeepHit/DeepHit/Output/weights/Adam_2*+
_class!
loc:@DeepHit/Output/weights_1*
dtype0
Ђ
IDeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros/shape_as_tensor_1Const*+
_class!
loc:@DeepHit/Output/weights_1*
dtype0*
valueB"(     
Щ
?DeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros/Const_1Const*+
_class!
loc:@DeepHit/Output/weights_1*
dtype0*
valueB
 *    
Х
9DeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros_1FillIDeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros/shape_as_tensor_1?DeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros/Const_1*
T0*+
_class!
loc:@DeepHit/Output/weights_1*

index_type0
ж
'DeepHit/DeepHit/Output/weights/Adam_1_1VarHandleOp*+
_class!
loc:@DeepHit/Output/weights_1*
allowed_devices
 *
	container *
dtype0*
shape:	(Ю*6
shared_name'%DeepHit/DeepHit/Output/weights/Adam_1
≤
FDeepHit/DeepHit/Output/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp'DeepHit/DeepHit/Output/weights/Adam_1_1*+
_class!
loc:@DeepHit/Output/weights_1
…
.DeepHit/DeepHit/Output/weights/Adam_1/Assign_1AssignVariableOp'DeepHit/DeepHit/Output/weights/Adam_1_19DeepHit/DeepHit/Output/weights/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
Ѓ
9DeepHit/DeepHit/Output/weights/Adam_1/Read/ReadVariableOpReadVariableOp'DeepHit/DeepHit/Output/weights/Adam_1_1*+
_class!
loc:@DeepHit/Output/weights_1*
dtype0
Ф
6DeepHit/DeepHit/Output/biases/Adam/Initializer/zeros_1Const**
_class 
loc:@DeepHit/Output/biases_1*
dtype0*
valueBЮ*    
џ
$DeepHit/DeepHit/Output/biases/Adam_2VarHandleOp**
_class 
loc:@DeepHit/Output/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:Ю*3
shared_name$"DeepHit/DeepHit/Output/biases/Adam
Ђ
CDeepHit/DeepHit/Output/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp$DeepHit/DeepHit/Output/biases/Adam_2**
_class 
loc:@DeepHit/Output/biases_1
ј
+DeepHit/DeepHit/Output/biases/Adam/Assign_1AssignVariableOp$DeepHit/DeepHit/Output/biases/Adam_26DeepHit/DeepHit/Output/biases/Adam/Initializer/zeros_1*
dtype0*
validate_shape( 
І
6DeepHit/DeepHit/Output/biases/Adam/Read/ReadVariableOpReadVariableOp$DeepHit/DeepHit/Output/biases/Adam_2**
_class 
loc:@DeepHit/Output/biases_1*
dtype0
Ц
8DeepHit/DeepHit/Output/biases/Adam_1/Initializer/zeros_1Const**
_class 
loc:@DeepHit/Output/biases_1*
dtype0*
valueBЮ*    
я
&DeepHit/DeepHit/Output/biases/Adam_1_1VarHandleOp**
_class 
loc:@DeepHit/Output/biases_1*
allowed_devices
 *
	container *
dtype0*
shape:Ю*5
shared_name&$DeepHit/DeepHit/Output/biases/Adam_1
ѓ
EDeepHit/DeepHit/Output/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp&DeepHit/DeepHit/Output/biases/Adam_1_1**
_class 
loc:@DeepHit/Output/biases_1
∆
-DeepHit/DeepHit/Output/biases/Adam_1/Assign_1AssignVariableOp&DeepHit/DeepHit/Output/biases/Adam_1_18DeepHit/DeepHit/Output/biases/Adam_1/Initializer/zeros_1*
dtype0*
validate_shape( 
Ђ
8DeepHit/DeepHit/Output/biases/Adam_1/Read/ReadVariableOpReadVariableOp&DeepHit/DeepHit/Output/biases/Adam_1_1**
_class 
loc:@DeepHit/Output/biases_1*
dtype0
A
DeepHit/Adam/beta1_1Const*
dtype0*
valueB
 *fff?
A
DeepHit/Adam/beta2_1Const*
dtype0*
valueB
 *wЊ?
C
DeepHit/Adam/epsilon_1Const*
dtype0*
valueB
 *wћ+2
К
TDeepHit/Adam/update_DeepHit/fully_connected/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
М
VDeepHit/Adam/update_DeepHit/fully_connected/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
м
EDeepHit/Adam/update_DeepHit/fully_connected/weights/ResourceApplyAdamResourceApplyAdam!DeepHit/fully_connected/weights_1.DeepHit/DeepHit/fully_connected/weights/Adam_20DeepHit/DeepHit/fully_connected/weights/Adam_1_1TDeepHit/Adam/update_DeepHit/fully_connected/weights/ResourceApplyAdam/ReadVariableOpVDeepHit/Adam/update_DeepHit/fully_connected/weights/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1DeepHit/gradients/AddN_9_1*
T0*4
_class*
(&loc:@DeepHit/fully_connected/weights_1*
use_locking( *
use_nesterov( 
Й
SDeepHit/Adam/update_DeepHit/fully_connected/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
Л
UDeepHit/Adam/update_DeepHit/fully_connected/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
Ю
DDeepHit/Adam/update_DeepHit/fully_connected/biases/ResourceApplyAdamResourceApplyAdam DeepHit/fully_connected/biases_1-DeepHit/DeepHit/fully_connected/biases/Adam_2/DeepHit/DeepHit/fully_connected/biases/Adam_1_1SDeepHit/Adam/update_DeepHit/fully_connected/biases/ResourceApplyAdam/ReadVariableOpUDeepHit/Adam/update_DeepHit/fully_connected/biases/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1SDeepHit/gradients/DeepHit/fully_connected/BiasAdd_grad/tuple/control_dependency_1_1*
T0*3
_class)
'%loc:@DeepHit/fully_connected/biases_1*
use_locking( *
use_nesterov( 
М
VDeepHit/Adam/update_DeepHit/fully_connected_1/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
О
XDeepHit/Adam/update_DeepHit/fully_connected_1/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
ъ
GDeepHit/Adam/update_DeepHit/fully_connected_1/weights/ResourceApplyAdamResourceApplyAdam#DeepHit/fully_connected_1/weights_10DeepHit/DeepHit/fully_connected_1/weights/Adam_22DeepHit/DeepHit/fully_connected_1/weights/Adam_1_1VDeepHit/Adam/update_DeepHit/fully_connected_1/weights/ResourceApplyAdam/ReadVariableOpXDeepHit/Adam/update_DeepHit/fully_connected_1/weights/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1DeepHit/gradients/AddN_8_1*
T0*6
_class,
*(loc:@DeepHit/fully_connected_1/weights_1*
use_locking( *
use_nesterov( 
Л
UDeepHit/Adam/update_DeepHit/fully_connected_1/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
Н
WDeepHit/Adam/update_DeepHit/fully_connected_1/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
Ѓ
FDeepHit/Adam/update_DeepHit/fully_connected_1/biases/ResourceApplyAdamResourceApplyAdam"DeepHit/fully_connected_1/biases_1/DeepHit/DeepHit/fully_connected_1/biases/Adam_21DeepHit/DeepHit/fully_connected_1/biases/Adam_1_1UDeepHit/Adam/update_DeepHit/fully_connected_1/biases/ResourceApplyAdam/ReadVariableOpWDeepHit/Adam/update_DeepHit/fully_connected_1/biases/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1UDeepHit/gradients/DeepHit/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1_1*
T0*5
_class+
)'loc:@DeepHit/fully_connected_1/biases_1*
use_locking( *
use_nesterov( 
М
VDeepHit/Adam/update_DeepHit/fully_connected_2/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
О
XDeepHit/Adam/update_DeepHit/fully_connected_2/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
ъ
GDeepHit/Adam/update_DeepHit/fully_connected_2/weights/ResourceApplyAdamResourceApplyAdam#DeepHit/fully_connected_2/weights_10DeepHit/DeepHit/fully_connected_2/weights/Adam_22DeepHit/DeepHit/fully_connected_2/weights/Adam_1_1VDeepHit/Adam/update_DeepHit/fully_connected_2/weights/ResourceApplyAdam/ReadVariableOpXDeepHit/Adam/update_DeepHit/fully_connected_2/weights/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1DeepHit/gradients/AddN_7_1*
T0*6
_class,
*(loc:@DeepHit/fully_connected_2/weights_1*
use_locking( *
use_nesterov( 
Л
UDeepHit/Adam/update_DeepHit/fully_connected_2/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
Н
WDeepHit/Adam/update_DeepHit/fully_connected_2/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
Ѓ
FDeepHit/Adam/update_DeepHit/fully_connected_2/biases/ResourceApplyAdamResourceApplyAdam"DeepHit/fully_connected_2/biases_1/DeepHit/DeepHit/fully_connected_2/biases/Adam_21DeepHit/DeepHit/fully_connected_2/biases/Adam_1_1UDeepHit/Adam/update_DeepHit/fully_connected_2/biases/ResourceApplyAdam/ReadVariableOpWDeepHit/Adam/update_DeepHit/fully_connected_2/biases/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1UDeepHit/gradients/DeepHit/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1_1*
T0*5
_class+
)'loc:@DeepHit/fully_connected_2/biases_1*
use_locking( *
use_nesterov( 
М
VDeepHit/Adam/update_DeepHit/fully_connected_3/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
О
XDeepHit/Adam/update_DeepHit/fully_connected_3/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
ъ
GDeepHit/Adam/update_DeepHit/fully_connected_3/weights/ResourceApplyAdamResourceApplyAdam#DeepHit/fully_connected_3/weights_10DeepHit/DeepHit/fully_connected_3/weights/Adam_22DeepHit/DeepHit/fully_connected_3/weights/Adam_1_1VDeepHit/Adam/update_DeepHit/fully_connected_3/weights/ResourceApplyAdam/ReadVariableOpXDeepHit/Adam/update_DeepHit/fully_connected_3/weights/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1DeepHit/gradients/AddN_5_1*
T0*6
_class,
*(loc:@DeepHit/fully_connected_3/weights_1*
use_locking( *
use_nesterov( 
Л
UDeepHit/Adam/update_DeepHit/fully_connected_3/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
Н
WDeepHit/Adam/update_DeepHit/fully_connected_3/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
Ѓ
FDeepHit/Adam/update_DeepHit/fully_connected_3/biases/ResourceApplyAdamResourceApplyAdam"DeepHit/fully_connected_3/biases_1/DeepHit/DeepHit/fully_connected_3/biases/Adam_21DeepHit/DeepHit/fully_connected_3/biases/Adam_1_1UDeepHit/Adam/update_DeepHit/fully_connected_3/biases/ResourceApplyAdam/ReadVariableOpWDeepHit/Adam/update_DeepHit/fully_connected_3/biases/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1UDeepHit/gradients/DeepHit/fully_connected_3/BiasAdd_grad/tuple/control_dependency_1_1*
T0*5
_class+
)'loc:@DeepHit/fully_connected_3/biases_1*
use_locking( *
use_nesterov( 
М
VDeepHit/Adam/update_DeepHit/fully_connected_4/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
О
XDeepHit/Adam/update_DeepHit/fully_connected_4/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
ъ
GDeepHit/Adam/update_DeepHit/fully_connected_4/weights/ResourceApplyAdamResourceApplyAdam#DeepHit/fully_connected_4/weights_10DeepHit/DeepHit/fully_connected_4/weights/Adam_22DeepHit/DeepHit/fully_connected_4/weights/Adam_1_1VDeepHit/Adam/update_DeepHit/fully_connected_4/weights/ResourceApplyAdam/ReadVariableOpXDeepHit/Adam/update_DeepHit/fully_connected_4/weights/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1DeepHit/gradients/AddN_6_1*
T0*6
_class,
*(loc:@DeepHit/fully_connected_4/weights_1*
use_locking( *
use_nesterov( 
Л
UDeepHit/Adam/update_DeepHit/fully_connected_4/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
Н
WDeepHit/Adam/update_DeepHit/fully_connected_4/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
Ѓ
FDeepHit/Adam/update_DeepHit/fully_connected_4/biases/ResourceApplyAdamResourceApplyAdam"DeepHit/fully_connected_4/biases_1/DeepHit/DeepHit/fully_connected_4/biases/Adam_21DeepHit/DeepHit/fully_connected_4/biases/Adam_1_1UDeepHit/Adam/update_DeepHit/fully_connected_4/biases/ResourceApplyAdam/ReadVariableOpWDeepHit/Adam/update_DeepHit/fully_connected_4/biases/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1UDeepHit/gradients/DeepHit/fully_connected_4/BiasAdd_grad/tuple/control_dependency_1_1*
T0*5
_class+
)'loc:@DeepHit/fully_connected_4/biases_1*
use_locking( *
use_nesterov( 
Б
KDeepHit/Adam/update_DeepHit/Output/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
Г
MDeepHit/Adam/update_DeepHit/Output/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
≠
<DeepHit/Adam/update_DeepHit/Output/weights/ResourceApplyAdamResourceApplyAdamDeepHit/Output/weights_1%DeepHit/DeepHit/Output/weights/Adam_2'DeepHit/DeepHit/Output/weights/Adam_1_1KDeepHit/Adam/update_DeepHit/Output/weights/ResourceApplyAdam/ReadVariableOpMDeepHit/Adam/update_DeepHit/Output/weights/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1DeepHit/gradients/AddN_3_1*
T0*+
_class!
loc:@DeepHit/Output/weights_1*
use_locking( *
use_nesterov( 
А
JDeepHit/Adam/update_DeepHit/Output/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1*
dtype0
В
LDeepHit/Adam/update_DeepHit/Output/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpDeepHit/beta2_power_1*
dtype0
÷
;DeepHit/Adam/update_DeepHit/Output/biases/ResourceApplyAdamResourceApplyAdamDeepHit/Output/biases_1$DeepHit/DeepHit/Output/biases/Adam_2&DeepHit/DeepHit/Output/biases/Adam_1_1JDeepHit/Adam/update_DeepHit/Output/biases/ResourceApplyAdam/ReadVariableOpLDeepHit/Adam/update_DeepHit/Output/biases/ResourceApplyAdam/ReadVariableOp_1DeepHit/learning_rate_1DeepHit/Adam/beta1_1DeepHit/Adam/beta2_1DeepHit/Adam/epsilon_1JDeepHit/gradients/DeepHit/Output/BiasAdd_grad/tuple/control_dependency_1_1*
T0**
_class 
loc:@DeepHit/Output/biases_1*
use_locking( *
use_nesterov( 
©
DeepHit/Adam/ReadVariableOpReadVariableOpDeepHit/beta1_power_1<^DeepHit/Adam/update_DeepHit/Output/biases/ResourceApplyAdam=^DeepHit/Adam/update_DeepHit/Output/weights/ResourceApplyAdamE^DeepHit/Adam/update_DeepHit/fully_connected/biases/ResourceApplyAdamF^DeepHit/Adam/update_DeepHit/fully_connected/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_1/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_1/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_2/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_2/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_3/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_3/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_4/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_4/weights/ResourceApplyAdam*
dtype0
Б
DeepHit/Adam/mul_2MulDeepHit/Adam/ReadVariableOpDeepHit/Adam/beta1_1*
T0**
_class 
loc:@DeepHit/Output/biases_1
Ђ
DeepHit/Adam/AssignVariableOpAssignVariableOpDeepHit/beta1_power_1DeepHit/Adam/mul_2**
_class 
loc:@DeepHit/Output/biases_1*
dtype0*
validate_shape( 
ч
DeepHit/Adam/ReadVariableOp_1ReadVariableOpDeepHit/beta1_power_1^DeepHit/Adam/AssignVariableOp<^DeepHit/Adam/update_DeepHit/Output/biases/ResourceApplyAdam=^DeepHit/Adam/update_DeepHit/Output/weights/ResourceApplyAdamE^DeepHit/Adam/update_DeepHit/fully_connected/biases/ResourceApplyAdamF^DeepHit/Adam/update_DeepHit/fully_connected/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_1/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_1/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_2/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_2/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_3/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_3/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_4/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_4/weights/ResourceApplyAdam**
_class 
loc:@DeepHit/Output/biases_1*
dtype0
Ђ
DeepHit/Adam/ReadVariableOp_2ReadVariableOpDeepHit/beta2_power_1<^DeepHit/Adam/update_DeepHit/Output/biases/ResourceApplyAdam=^DeepHit/Adam/update_DeepHit/Output/weights/ResourceApplyAdamE^DeepHit/Adam/update_DeepHit/fully_connected/biases/ResourceApplyAdamF^DeepHit/Adam/update_DeepHit/fully_connected/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_1/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_1/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_2/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_2/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_3/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_3/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_4/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_4/weights/ResourceApplyAdam*
dtype0
Е
DeepHit/Adam/mul_1_1MulDeepHit/Adam/ReadVariableOp_2DeepHit/Adam/beta2_1*
T0**
_class 
loc:@DeepHit/Output/biases_1
ѓ
DeepHit/Adam/AssignVariableOp_1AssignVariableOpDeepHit/beta2_power_1DeepHit/Adam/mul_1_1**
_class 
loc:@DeepHit/Output/biases_1*
dtype0*
validate_shape( 
щ
DeepHit/Adam/ReadVariableOp_3ReadVariableOpDeepHit/beta2_power_1 ^DeepHit/Adam/AssignVariableOp_1<^DeepHit/Adam/update_DeepHit/Output/biases/ResourceApplyAdam=^DeepHit/Adam/update_DeepHit/Output/weights/ResourceApplyAdamE^DeepHit/Adam/update_DeepHit/fully_connected/biases/ResourceApplyAdamF^DeepHit/Adam/update_DeepHit/fully_connected/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_1/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_1/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_2/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_2/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_3/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_3/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_4/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_4/weights/ResourceApplyAdam**
_class 
loc:@DeepHit/Output/biases_1*
dtype0
∞
DeepHit/Adam_1NoOp^DeepHit/Adam/AssignVariableOp ^DeepHit/Adam/AssignVariableOp_1<^DeepHit/Adam/update_DeepHit/Output/biases/ResourceApplyAdam=^DeepHit/Adam/update_DeepHit/Output/weights/ResourceApplyAdamE^DeepHit/Adam/update_DeepHit/fully_connected/biases/ResourceApplyAdamF^DeepHit/Adam/update_DeepHit/fully_connected/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_1/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_1/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_2/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_2/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_3/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_3/weights/ResourceApplyAdamG^DeepHit/Adam/update_DeepHit/fully_connected_4/biases/ResourceApplyAdamH^DeepHit/Adam/update_DeepHit/fully_connected_4/weights/ResourceApplyAdam
C
save/filename/input_1Const*
dtype0*
valueB Bmodel
Z
save/filename_1PlaceholderWithDefaultsave/filename/input_1*
dtype0*
shape: 
Q
save/Const_1PlaceholderWithDefaultsave/filename_1*
dtype0*
shape: 
ќ
save/SaveV2/tensor_names_1Const*
dtype0*Ы
valueСBО&B"DeepHit/DeepHit/Output/biases/AdamB$DeepHit/DeepHit/Output/biases/Adam_1B#DeepHit/DeepHit/Output/weights/AdamB%DeepHit/DeepHit/Output/weights/Adam_1B+DeepHit/DeepHit/fully_connected/biases/AdamB-DeepHit/DeepHit/fully_connected/biases/Adam_1B,DeepHit/DeepHit/fully_connected/weights/AdamB.DeepHit/DeepHit/fully_connected/weights/Adam_1B-DeepHit/DeepHit/fully_connected_1/biases/AdamB/DeepHit/DeepHit/fully_connected_1/biases/Adam_1B.DeepHit/DeepHit/fully_connected_1/weights/AdamB0DeepHit/DeepHit/fully_connected_1/weights/Adam_1B-DeepHit/DeepHit/fully_connected_2/biases/AdamB/DeepHit/DeepHit/fully_connected_2/biases/Adam_1B.DeepHit/DeepHit/fully_connected_2/weights/AdamB0DeepHit/DeepHit/fully_connected_2/weights/Adam_1B-DeepHit/DeepHit/fully_connected_3/biases/AdamB/DeepHit/DeepHit/fully_connected_3/biases/Adam_1B.DeepHit/DeepHit/fully_connected_3/weights/AdamB0DeepHit/DeepHit/fully_connected_3/weights/Adam_1B-DeepHit/DeepHit/fully_connected_4/biases/AdamB/DeepHit/DeepHit/fully_connected_4/biases/Adam_1B.DeepHit/DeepHit/fully_connected_4/weights/AdamB0DeepHit/DeepHit/fully_connected_4/weights/Adam_1BDeepHit/Output/biasesBDeepHit/Output/weightsBDeepHit/beta1_powerBDeepHit/beta2_powerBDeepHit/fully_connected/biasesBDeepHit/fully_connected/weightsB DeepHit/fully_connected_1/biasesB!DeepHit/fully_connected_1/weightsB DeepHit/fully_connected_2/biasesB!DeepHit/fully_connected_2/weightsB DeepHit/fully_connected_3/biasesB!DeepHit/fully_connected_3/weightsB DeepHit/fully_connected_4/biasesB!DeepHit/fully_connected_4/weights
Х
save/SaveV2/shape_and_slices_1Const*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Х
save/SaveV2_1SaveV2save/Const_1save/SaveV2/tensor_names_1save/SaveV2/shape_and_slices_16DeepHit/DeepHit/Output/biases/Adam/Read/ReadVariableOp8DeepHit/DeepHit/Output/biases/Adam_1/Read/ReadVariableOp7DeepHit/DeepHit/Output/weights/Adam/Read/ReadVariableOp9DeepHit/DeepHit/Output/weights/Adam_1/Read/ReadVariableOp?DeepHit/DeepHit/fully_connected/biases/Adam/Read/ReadVariableOpADeepHit/DeepHit/fully_connected/biases/Adam_1/Read/ReadVariableOp@DeepHit/DeepHit/fully_connected/weights/Adam/Read/ReadVariableOpBDeepHit/DeepHit/fully_connected/weights/Adam_1/Read/ReadVariableOpADeepHit/DeepHit/fully_connected_1/biases/Adam/Read/ReadVariableOpCDeepHit/DeepHit/fully_connected_1/biases/Adam_1/Read/ReadVariableOpBDeepHit/DeepHit/fully_connected_1/weights/Adam/Read/ReadVariableOpDDeepHit/DeepHit/fully_connected_1/weights/Adam_1/Read/ReadVariableOpADeepHit/DeepHit/fully_connected_2/biases/Adam/Read/ReadVariableOpCDeepHit/DeepHit/fully_connected_2/biases/Adam_1/Read/ReadVariableOpBDeepHit/DeepHit/fully_connected_2/weights/Adam/Read/ReadVariableOpDDeepHit/DeepHit/fully_connected_2/weights/Adam_1/Read/ReadVariableOpADeepHit/DeepHit/fully_connected_3/biases/Adam/Read/ReadVariableOpCDeepHit/DeepHit/fully_connected_3/biases/Adam_1/Read/ReadVariableOpBDeepHit/DeepHit/fully_connected_3/weights/Adam/Read/ReadVariableOpDDeepHit/DeepHit/fully_connected_3/weights/Adam_1/Read/ReadVariableOpADeepHit/DeepHit/fully_connected_4/biases/Adam/Read/ReadVariableOpCDeepHit/DeepHit/fully_connected_4/biases/Adam_1/Read/ReadVariableOpBDeepHit/DeepHit/fully_connected_4/weights/Adam/Read/ReadVariableOpDDeepHit/DeepHit/fully_connected_4/weights/Adam_1/Read/ReadVariableOp)DeepHit/Output/biases/Read/ReadVariableOp*DeepHit/Output/weights/Read/ReadVariableOp'DeepHit/beta1_power/Read/ReadVariableOp'DeepHit/beta2_power/Read/ReadVariableOp2DeepHit/fully_connected/biases/Read/ReadVariableOp3DeepHit/fully_connected/weights/Read/ReadVariableOp4DeepHit/fully_connected_1/biases/Read/ReadVariableOp5DeepHit/fully_connected_1/weights/Read/ReadVariableOp4DeepHit/fully_connected_2/biases/Read/ReadVariableOp5DeepHit/fully_connected_2/weights/Read/ReadVariableOp4DeepHit/fully_connected_3/biases/Read/ReadVariableOp5DeepHit/fully_connected_3/weights/Read/ReadVariableOp4DeepHit/fully_connected_4/biases/Read/ReadVariableOp5DeepHit/fully_connected_4/weights/Read/ReadVariableOp*4
dtypes*
(2&
m
save/control_dependency_1Identitysave/Const_1^save/SaveV2_1*
T0*
_class
loc:@save/Const_1
а
save/RestoreV2/tensor_names_1Const"/device:CPU:0*
dtype0*Ы
valueСBО&B"DeepHit/DeepHit/Output/biases/AdamB$DeepHit/DeepHit/Output/biases/Adam_1B#DeepHit/DeepHit/Output/weights/AdamB%DeepHit/DeepHit/Output/weights/Adam_1B+DeepHit/DeepHit/fully_connected/biases/AdamB-DeepHit/DeepHit/fully_connected/biases/Adam_1B,DeepHit/DeepHit/fully_connected/weights/AdamB.DeepHit/DeepHit/fully_connected/weights/Adam_1B-DeepHit/DeepHit/fully_connected_1/biases/AdamB/DeepHit/DeepHit/fully_connected_1/biases/Adam_1B.DeepHit/DeepHit/fully_connected_1/weights/AdamB0DeepHit/DeepHit/fully_connected_1/weights/Adam_1B-DeepHit/DeepHit/fully_connected_2/biases/AdamB/DeepHit/DeepHit/fully_connected_2/biases/Adam_1B.DeepHit/DeepHit/fully_connected_2/weights/AdamB0DeepHit/DeepHit/fully_connected_2/weights/Adam_1B-DeepHit/DeepHit/fully_connected_3/biases/AdamB/DeepHit/DeepHit/fully_connected_3/biases/Adam_1B.DeepHit/DeepHit/fully_connected_3/weights/AdamB0DeepHit/DeepHit/fully_connected_3/weights/Adam_1B-DeepHit/DeepHit/fully_connected_4/biases/AdamB/DeepHit/DeepHit/fully_connected_4/biases/Adam_1B.DeepHit/DeepHit/fully_connected_4/weights/AdamB0DeepHit/DeepHit/fully_connected_4/weights/Adam_1BDeepHit/Output/biasesBDeepHit/Output/weightsBDeepHit/beta1_powerBDeepHit/beta2_powerBDeepHit/fully_connected/biasesBDeepHit/fully_connected/weightsB DeepHit/fully_connected_1/biasesB!DeepHit/fully_connected_1/weightsB DeepHit/fully_connected_2/biasesB!DeepHit/fully_connected_2/weightsB DeepHit/fully_connected_3/biasesB!DeepHit/fully_connected_3/weightsB DeepHit/fully_connected_4/biasesB!DeepHit/fully_connected_4/weights
І
!save/RestoreV2/shape_and_slices_1Const"/device:CPU:0*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
≤
save/RestoreV2_1	RestoreV2save/Const_1save/RestoreV2/tensor_names_1!save/RestoreV2/shape_and_slices_1"/device:CPU:0*4
dtypes*
(2&
4
save/IdentityIdentitysave/RestoreV2_1*
T0
Б
save/AssignVariableOpAssignVariableOp$DeepHit/DeepHit/Output/biases/Adam_2save/Identity*
dtype0*
validate_shape( 
8
save/Identity_1Identitysave/RestoreV2_1:1*
T0
З
save/AssignVariableOp_1AssignVariableOp&DeepHit/DeepHit/Output/biases/Adam_1_1save/Identity_1*
dtype0*
validate_shape( 
8
save/Identity_2Identitysave/RestoreV2_1:2*
T0
Ж
save/AssignVariableOp_2AssignVariableOp%DeepHit/DeepHit/Output/weights/Adam_2save/Identity_2*
dtype0*
validate_shape( 
8
save/Identity_3Identitysave/RestoreV2_1:3*
T0
И
save/AssignVariableOp_3AssignVariableOp'DeepHit/DeepHit/Output/weights/Adam_1_1save/Identity_3*
dtype0*
validate_shape( 
8
save/Identity_4Identitysave/RestoreV2_1:4*
T0
О
save/AssignVariableOp_4AssignVariableOp-DeepHit/DeepHit/fully_connected/biases/Adam_2save/Identity_4*
dtype0*
validate_shape( 
8
save/Identity_5Identitysave/RestoreV2_1:5*
T0
Р
save/AssignVariableOp_5AssignVariableOp/DeepHit/DeepHit/fully_connected/biases/Adam_1_1save/Identity_5*
dtype0*
validate_shape( 
8
save/Identity_6Identitysave/RestoreV2_1:6*
T0
П
save/AssignVariableOp_6AssignVariableOp.DeepHit/DeepHit/fully_connected/weights/Adam_2save/Identity_6*
dtype0*
validate_shape( 
8
save/Identity_7Identitysave/RestoreV2_1:7*
T0
С
save/AssignVariableOp_7AssignVariableOp0DeepHit/DeepHit/fully_connected/weights/Adam_1_1save/Identity_7*
dtype0*
validate_shape( 
8
save/Identity_8Identitysave/RestoreV2_1:8*
T0
Р
save/AssignVariableOp_8AssignVariableOp/DeepHit/DeepHit/fully_connected_1/biases/Adam_2save/Identity_8*
dtype0*
validate_shape( 
8
save/Identity_9Identitysave/RestoreV2_1:9*
T0
Т
save/AssignVariableOp_9AssignVariableOp1DeepHit/DeepHit/fully_connected_1/biases/Adam_1_1save/Identity_9*
dtype0*
validate_shape( 
:
save/Identity_10Identitysave/RestoreV2_1:10*
T0
У
save/AssignVariableOp_10AssignVariableOp0DeepHit/DeepHit/fully_connected_1/weights/Adam_2save/Identity_10*
dtype0*
validate_shape( 
:
save/Identity_11Identitysave/RestoreV2_1:11*
T0
Х
save/AssignVariableOp_11AssignVariableOp2DeepHit/DeepHit/fully_connected_1/weights/Adam_1_1save/Identity_11*
dtype0*
validate_shape( 
:
save/Identity_12Identitysave/RestoreV2_1:12*
T0
Т
save/AssignVariableOp_12AssignVariableOp/DeepHit/DeepHit/fully_connected_2/biases/Adam_2save/Identity_12*
dtype0*
validate_shape( 
:
save/Identity_13Identitysave/RestoreV2_1:13*
T0
Ф
save/AssignVariableOp_13AssignVariableOp1DeepHit/DeepHit/fully_connected_2/biases/Adam_1_1save/Identity_13*
dtype0*
validate_shape( 
:
save/Identity_14Identitysave/RestoreV2_1:14*
T0
У
save/AssignVariableOp_14AssignVariableOp0DeepHit/DeepHit/fully_connected_2/weights/Adam_2save/Identity_14*
dtype0*
validate_shape( 
:
save/Identity_15Identitysave/RestoreV2_1:15*
T0
Х
save/AssignVariableOp_15AssignVariableOp2DeepHit/DeepHit/fully_connected_2/weights/Adam_1_1save/Identity_15*
dtype0*
validate_shape( 
:
save/Identity_16Identitysave/RestoreV2_1:16*
T0
Т
save/AssignVariableOp_16AssignVariableOp/DeepHit/DeepHit/fully_connected_3/biases/Adam_2save/Identity_16*
dtype0*
validate_shape( 
:
save/Identity_17Identitysave/RestoreV2_1:17*
T0
Ф
save/AssignVariableOp_17AssignVariableOp1DeepHit/DeepHit/fully_connected_3/biases/Adam_1_1save/Identity_17*
dtype0*
validate_shape( 
:
save/Identity_18Identitysave/RestoreV2_1:18*
T0
У
save/AssignVariableOp_18AssignVariableOp0DeepHit/DeepHit/fully_connected_3/weights/Adam_2save/Identity_18*
dtype0*
validate_shape( 
:
save/Identity_19Identitysave/RestoreV2_1:19*
T0
Х
save/AssignVariableOp_19AssignVariableOp2DeepHit/DeepHit/fully_connected_3/weights/Adam_1_1save/Identity_19*
dtype0*
validate_shape( 
:
save/Identity_20Identitysave/RestoreV2_1:20*
T0
Т
save/AssignVariableOp_20AssignVariableOp/DeepHit/DeepHit/fully_connected_4/biases/Adam_2save/Identity_20*
dtype0*
validate_shape( 
:
save/Identity_21Identitysave/RestoreV2_1:21*
T0
Ф
save/AssignVariableOp_21AssignVariableOp1DeepHit/DeepHit/fully_connected_4/biases/Adam_1_1save/Identity_21*
dtype0*
validate_shape( 
:
save/Identity_22Identitysave/RestoreV2_1:22*
T0
У
save/AssignVariableOp_22AssignVariableOp0DeepHit/DeepHit/fully_connected_4/weights/Adam_2save/Identity_22*
dtype0*
validate_shape( 
:
save/Identity_23Identitysave/RestoreV2_1:23*
T0
Х
save/AssignVariableOp_23AssignVariableOp2DeepHit/DeepHit/fully_connected_4/weights/Adam_1_1save/Identity_23*
dtype0*
validate_shape( 
:
save/Identity_24Identitysave/RestoreV2_1:24*
T0
z
save/AssignVariableOp_24AssignVariableOpDeepHit/Output/biases_1save/Identity_24*
dtype0*
validate_shape( 
:
save/Identity_25Identitysave/RestoreV2_1:25*
T0
{
save/AssignVariableOp_25AssignVariableOpDeepHit/Output/weights_1save/Identity_25*
dtype0*
validate_shape( 
:
save/Identity_26Identitysave/RestoreV2_1:26*
T0
x
save/AssignVariableOp_26AssignVariableOpDeepHit/beta1_power_1save/Identity_26*
dtype0*
validate_shape( 
:
save/Identity_27Identitysave/RestoreV2_1:27*
T0
x
save/AssignVariableOp_27AssignVariableOpDeepHit/beta2_power_1save/Identity_27*
dtype0*
validate_shape( 
:
save/Identity_28Identitysave/RestoreV2_1:28*
T0
Г
save/AssignVariableOp_28AssignVariableOp DeepHit/fully_connected/biases_1save/Identity_28*
dtype0*
validate_shape( 
:
save/Identity_29Identitysave/RestoreV2_1:29*
T0
Д
save/AssignVariableOp_29AssignVariableOp!DeepHit/fully_connected/weights_1save/Identity_29*
dtype0*
validate_shape( 
:
save/Identity_30Identitysave/RestoreV2_1:30*
T0
Е
save/AssignVariableOp_30AssignVariableOp"DeepHit/fully_connected_1/biases_1save/Identity_30*
dtype0*
validate_shape( 
:
save/Identity_31Identitysave/RestoreV2_1:31*
T0
Ж
save/AssignVariableOp_31AssignVariableOp#DeepHit/fully_connected_1/weights_1save/Identity_31*
dtype0*
validate_shape( 
:
save/Identity_32Identitysave/RestoreV2_1:32*
T0
Е
save/AssignVariableOp_32AssignVariableOp"DeepHit/fully_connected_2/biases_1save/Identity_32*
dtype0*
validate_shape( 
:
save/Identity_33Identitysave/RestoreV2_1:33*
T0
Ж
save/AssignVariableOp_33AssignVariableOp#DeepHit/fully_connected_2/weights_1save/Identity_33*
dtype0*
validate_shape( 
:
save/Identity_34Identitysave/RestoreV2_1:34*
T0
Е
save/AssignVariableOp_34AssignVariableOp"DeepHit/fully_connected_3/biases_1save/Identity_34*
dtype0*
validate_shape( 
:
save/Identity_35Identitysave/RestoreV2_1:35*
T0
Ж
save/AssignVariableOp_35AssignVariableOp#DeepHit/fully_connected_3/weights_1save/Identity_35*
dtype0*
validate_shape( 
:
save/Identity_36Identitysave/RestoreV2_1:36*
T0
Е
save/AssignVariableOp_36AssignVariableOp"DeepHit/fully_connected_4/biases_1save/Identity_36*
dtype0*
validate_shape( 
:
save/Identity_37Identitysave/RestoreV2_1:37*
T0
Ж
save/AssignVariableOp_37AssignVariableOp#DeepHit/fully_connected_4/weights_1save/Identity_37*
dtype0*
validate_shape( 
Р
save/restore_all_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
Р
init_1NoOp,^DeepHit/DeepHit/Output/biases/Adam/Assign_1.^DeepHit/DeepHit/Output/biases/Adam_1/Assign_1-^DeepHit/DeepHit/Output/weights/Adam/Assign_1/^DeepHit/DeepHit/Output/weights/Adam_1/Assign_15^DeepHit/DeepHit/fully_connected/biases/Adam/Assign_17^DeepHit/DeepHit/fully_connected/biases/Adam_1/Assign_16^DeepHit/DeepHit/fully_connected/weights/Adam/Assign_18^DeepHit/DeepHit/fully_connected/weights/Adam_1/Assign_17^DeepHit/DeepHit/fully_connected_1/biases/Adam/Assign_19^DeepHit/DeepHit/fully_connected_1/biases/Adam_1/Assign_18^DeepHit/DeepHit/fully_connected_1/weights/Adam/Assign_1:^DeepHit/DeepHit/fully_connected_1/weights/Adam_1/Assign_17^DeepHit/DeepHit/fully_connected_2/biases/Adam/Assign_19^DeepHit/DeepHit/fully_connected_2/biases/Adam_1/Assign_18^DeepHit/DeepHit/fully_connected_2/weights/Adam/Assign_1:^DeepHit/DeepHit/fully_connected_2/weights/Adam_1/Assign_17^DeepHit/DeepHit/fully_connected_3/biases/Adam/Assign_19^DeepHit/DeepHit/fully_connected_3/biases/Adam_1/Assign_18^DeepHit/DeepHit/fully_connected_3/weights/Adam/Assign_1:^DeepHit/DeepHit/fully_connected_3/weights/Adam_1/Assign_17^DeepHit/DeepHit/fully_connected_4/biases/Adam/Assign_19^DeepHit/DeepHit/fully_connected_4/biases/Adam_1/Assign_18^DeepHit/DeepHit/fully_connected_4/weights/Adam/Assign_1:^DeepHit/DeepHit/fully_connected_4/weights/Adam_1/Assign_1^DeepHit/Output/biases/Assign_1 ^DeepHit/Output/weights/Assign_1^DeepHit/beta1_power/Assign_1^DeepHit/beta2_power/Assign_1(^DeepHit/fully_connected/biases/Assign_1)^DeepHit/fully_connected/weights/Assign_1*^DeepHit/fully_connected_1/biases/Assign_1+^DeepHit/fully_connected_1/weights/Assign_1*^DeepHit/fully_connected_2/biases/Assign_1+^DeepHit/fully_connected_2/weights/Assign_1*^DeepHit/fully_connected_3/biases/Assign_1+^DeepHit/fully_connected_3/weights/Assign_1*^DeepHit/fully_connected_4/biases/Assign_1+^DeepHit/fully_connected_4/weights/Assign_1"џ