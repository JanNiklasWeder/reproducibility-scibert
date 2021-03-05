using Flux
using Flux: onehot, onecold
using Transformers
using Transformers.Basic

labels = collect(1:10)
startsym = 11
endsym = 12
unksym = 0
labels = [unksym, startsym, endsym, labels...]
vocab = Vocabulary(labels, unksym)

#function for generate training datas
sample_data() = (d = rand(1:10, 10); (d,d))
#function for adding start & end symbol
preprocess(x) = [startsym, x..., endsym]

@show sample = preprocess.(sample_data())
@show encoded_sample = vocab(sample[1]) #use Vocabulary to encode the training data


#define a Word embedding layer which turn word index to word vector
embed = Embed(512, length(vocab))
#define a position embedding layer metioned above
pe = PositionEmbedding(512)

#wrapper for get embedding
function embedding(x)
  we = embed(x, inv(sqrt(512)))
  e = we .+ pe(we)
	return e
end

#define 2 layer of transformer
encode_t1 = Transformer(512, 8, 64, 2048)
encode_t2 = Transformer(512, 8, 64, 2048)

#define 2 layer of transformer decoder
decode_t1 = TransformerDecoder(512, 8, 64, 2048)
decode_t2 = TransformerDecoder(512, 8, 64, 2048)

#define the layer to get the final output probabilities
linear = Positionwise(Dense(512, length(vocab)), logsoftmax)

function encoder_forward(x)
  e = embedding(x)
  t1 = encode_t1(e)
  t2 = encode_t2(t1)
  return t2
end

function decoder_forward(x, m)
  e = embedding(x)
  t1 = decode_t1(e, m)
  t2 = decode_t2(t1, m)
  p = linear(t2)
	return p
end

enc = encoder_forward(encoded_sample)
probs = decoder_forward(encoded_sample, enc)

function smooth(et)
    sm = fill!(similar(et, Float32), 1e-6/size(embed, 2))
    p = sm .* (1 .+ -et)
    label = p .+ et .* (1 - convert(Float32, 1e-6))
    label
end

#define loss function
function loss(x, y)
  label = onehot(vocab, y) #turn the index to one-hot encoding
  label = smooth(label) #perform label smoothing
  enc = encoder_forward(x)
	probs = decoder_forward(y, enc)
  l = logkldivergence(label[:, 2:end, :], probs[:, 1:end-1, :])
  return l
end

#collect all the parameters
ps = params(embed, pe, encode_t1, encode_t2, decode_t1, decode_t2, linear)
opt = ADAM(1e-4)

#function for created batched data
using Transformers.Datasets: batched

#flux function for update parameters
using Flux: gradient
using Flux.Optimise: update!

#define training loop
function train!()
  @info "start training"
  for i = 1:2000
    data = batched([sample_data() for i = 1:32]) #create 32 random sample and batched
		x, y = preprocess.(data[1]), preprocess.(data[2])
    x, y = vocab(x), vocab(y)#encode the data
    x, y = x, y #move to gpu
    l = loss(x, y)
    grad = gradient(()->l, ps)
    if i % 8 == 0
    	println("loss = $l")
    end
    update!(opt, ps, grad)
  end
end


train!()


using Flux: onecold
function translate(x)
    ix = vocab(preprocess(x))
    seq = [startsym]

    enc = encoder_forward(ix)

    len = length(ix)
    for i = 1:2len
        trg = vocab(seq)
        dec = decoder_forward(trg, enc)
        #move back to gpu due to argmax wrong result on CuArrays
        ntok = onecold(collect(dec), labels)
        push!(seq, ntok[end])
        ntok[end] == endsym && break
    end
  seq[2:end-1]
end

translate([5,5,6,6,1,2,3,4,7, 10])

#example by Peter Cheng vgl. https://nextjournal.com/chengchingwen/jsoc-2019-blog2end-of-phase-one-hows-bert-going
