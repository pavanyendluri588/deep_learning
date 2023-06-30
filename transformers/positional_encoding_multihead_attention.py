import numpy as np 
#input embedding mateix 
#length of the input_length like no of columsn or no of words 
input_length=17
#length of the input embedding 
input_embedding_length=5
#craeting the input embedding matrix 
input_embedding_matrix = np.ones((input_length,input_embedding_length),dtype="float")
#print("input_embedding_matrix:\n",input_embedding_matrix)


#defining the positional encoding dimensions 
positional_encoding_dimension=10
#creating the positional encoding maatrix 
positional_encoding_matrix=np.zeros((input_length,positional_encoding_dimension),dtype="float")
#print(positional_encoding_matrix)
#inserting the values into the positional encoding matrix 
for input_position in range(input_length):
    for positional_encoding_dimension_index in range(positional_encoding_dimension):
        if positional_encoding_dimension_index%2==0:
            positional_encoding_matrix[input_position,positional_encoding_dimension_index]= np.sin(input_position/(10000**(positional_encoding_dimension_index/positional_encoding_dimension)))
        else:
            positional_encoding_matrix[input_position,positional_encoding_dimension_index]=np.cos(input_position/(10000**((positional_encoding_dimension_index-1)/positional_encoding_dimension)))
#print(positional_encoding_matrix)

#creating the matrix with the input and postion matrix combined
input_and_postioned_matrix=np.zeros((input_length,input_embedding_length+positional_encoding_dimension),dtype="float")
#print("input_embedding_matrix.shape:",input_embedding_matrix.shape,"input_and_postioned_matrix.shape:",input_and_postioned_matrix.shape)

#transpozing the positional_encoding_matrix
positional_encoding_matrix.transpose()
#print(positional_encoding_matrix.shape)
for input_index in range(input_length):
    #input_and_postioned_matrix[input_index]=input_embedding_matrix[input_index] + positional_encoding_matrix[input_index]
    input_and_postioned_matrix[input_index]=np.array(list(input_embedding_matrix[input_index])+list(positional_encoding_matrix[input_index]))
#print("input_and_postioned_matrix.shape:",input_and_postioned_matrix.shape)
#print("input_and_postioned_matrix:",input_and_postioned_matrix)

#programming the multi head attention

#programming the linerar layer 
 #no of output weights for linear layer 
no_of_output_weights=5
query_linear_layer_weights=np.random.rand(positional_encoding_dimension+input_embedding_length,no_of_output_weights)
key_linear_layer_weights=np.random.rand(positional_encoding_dimension+input_embedding_length,no_of_output_weights)
value_linear_layer_weights=np.random.rand(positional_encoding_dimension+input_embedding_length,no_of_output_weights)
#print("query_linear_layer_weights:",query_linear_layer_weights,"\nkey_linear_layer_weights:",key_linear_layer_weights,"\n key_linear_layer_weights:",key_linear_layer_weights)

#print(positional_encoding_matrix.shape,query_linear_layer_weights.shape)
#print(np.dot(positional_encoding_matrix,query_linear_layer_weights))
query_linear_layer_output=np.dot(input_and_postioned_matrix,query_linear_layer_weights)
key_linear_layer_output=np.dot(input_and_postioned_matrix,key_linear_layer_weights)
value_linear_layer_output=np.dot(input_and_postioned_matrix,value_linear_layer_weights)
#matmul layer means matrix multplication 
key_linear_layer_output= np.transpose(key_linear_layer_output)
#print(query_linear_layer_output.shape,key_linear_layer_output.shape)
mut_mul_layer_output=np.matmul(query_linear_layer_output,key_linear_layer_output)
print("mut_mul_layer_output:\n",mut_mul_layer_output)

#programming the scalar layer 
print(mut_mul_layer_output.shape[0])
print("mut_mul_layer_output/np.sqrt(mut_mul_layer_output.shape[0]):]\n",mut_mul_layer_output/np.sqrt(input_length))
scalar_layer_output=mut_mul_layer_output/np.sqrt(input_length)
print("scalar_layer_output:\n",scalar_layer_output)

#programming the softmax layer 
def softmax(x):
    experssion = np.exp(x)
    print("experssion:\n",experssion)
    sum_of_expressions=np.sum(experssion,axis=1)
    print("sum_of_expressions:\n",sum_of_expressions)
    sum_of_expressions_keepdims=np.sum(experssion,axis=1,keepdims=True)
    print("sum_of_expressions_keepdims:\n",sum_of_expressions_keepdims)
    return experssion/sum_of_expressions_keepdims
softmax_layer_output=softmax(scalar_layer_output)
print("softmax_layer_output:\n",softmax_layer_output)



#creating the complete head attention in single function 
#each single head attenction can recognize the single feature in main project we use the multiple heads to recognize the multiple features 
def head_attention(positional_encoding_matrix,no_of_resultant_output_weights_for_linear_layer=15):
    """
    \n
    positional_encoding_matrix rows = no of words  columns = (word embedding length + positionsal encoding dimension)
    shape of positional_encoding_matrix =  (no of words)*(word embedding length + positionsal encoding dimension)
    ex:
    positional_encoding_matrix
    sentence="hi this is pavan i am a quick learner"
    rows = no of words = 7



    use no_of_resultant_output_weights_for_linear_layer for creating the queru linear layer weights 
    shape of no_of_output_weights_for_linear_layer = (word embedding length + positionsal encoding dimension)*no_of_resultant_output_weights_for_linear_layer
    these weights are updated while trainiing 



    shape of output matrix (no of wods)*(no_of_resultant_output_weights_for_linear_layer)

                                                         

    """
    positional_encoding_dimension=positional_encoding_matrix.shape[1]
    query_linear_layer_weights=np.random.rand(positional_encoding_dimension,no_of_resultant_output_weights_for_linear_layer)
    key_linear_layer_weights=np.random.rand(positional_encoding_dimension,no_of_resultant_output_weights_for_linear_layer)
    value_linear_layer_weights=np.random.rand(positional_encoding_dimension,no_of_resultant_output_weights_for_linear_layer)

    query_linear_layer_output=np.dot(positional_encoding_matrix,query_linear_layer_weights)
    key_linear_layer_output=np.dot(positional_encoding_matrix,key_linear_layer_weights)
    value_linear_layer_output=np.dot(positional_encoding_matrix,value_linear_layer_weights)
    
    




    print("positional_encoding_matrix:",positional_encoding_matrix.shape,"query_linear_layer_weights:",query_linear_layer_weights.shape,"query_linear_layer_output:",query_linear_layer_output.shape)
    key_linear_layer_output_transposed=np.transpose(key_linear_layer_output)
    print("key_linear_layer_output_transposed:",key_linear_layer_output_transposed.shape)

    #creating the mat nul later 
    matmul_layer_output=np.matmul(query_linear_layer_output,key_linear_layer_output_transposed)
    print("matmul_layer_output",matmul_layer_output.shape)
    print("matmul_layer_output:\n",matmul_layer_output)

    #creating the scale layer 
    scale_layer_output=matmul_layer_output/np.sqrt(matmul_layer_output.shape[1])
    print("scale_layer_output:\n",scale_layer_output)

    #creating the softmax layer 
    expression_matrix=np.exp(scale_layer_output)
    sum_of_exprission_matrix_column_wise=np.sum(expression_matrix,axis=1,keepdims=True)
    softmax_layer_output=expression_matrix/sum_of_exprission_matrix_column_wise
    print("softmax_layer_output:\n",softmax_layer_output)

    #creating mat mul laayer
    matnul_layer_output_2=np.matmul(softmax_layer_output,value_linear_layer_output)
    print("matnul_layer_output_2:",matnul_layer_output_2.shape)

    return matnul_layer_output_2




head_attention1_output=head_attention(positional_encoding_matrix=input_and_postioned_matrix)
print("head_attention1_output:\n",head_attention1_output)
head_attention1_output=head_attention(positional_encoding_matrix=input_and_postioned_matrix)
head_attention1_output=head_attention(positional_encoding_matrix=input_and_postioned_matrix)

help(head_attention)