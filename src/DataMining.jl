using Pkg
Pkg.add("Flux")
Pkg.add("Transformers")

#using CUDA
using Flux
using Transformers
using Transformers.Basic
using Transformers.Pretrain

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

text1 = "Peter Piper picked a peck of pickled peppers" |> tokenizer |> wordpiece
text2 = "Fuzzy Wuzzy was a bear" |> tokenizer |> wordpiece

textArray = ["Peter Piper picked a peck of pickled peppers","Fuzzy Wuzzy was a bear"]

text = ["[CLS]"; text1; "[SEP]"; text2; "[SEP]"]
textCpoy = wordpiece.(tokenizer.(textArray))

@assert text == [
    "[CLS]", "peter", "piper", "picked", "a", "peck", "of", "pick", "##led", "peppers", "[SEP]",
    "fuzzy", "wu", "##zzy",  "was", "a", "bear", "[SEP]"
]

token_indices = vocab(text)
token_indices2 = vocab(textCpoy)
#vocab = Transformers.Basic.Vocabulary(wordpiece.vocab, wordpiece.vocab[wordpiece.unk_idx])

segment_indices = [fill(1, length(text1)+2); fill(2, length(text2)+1)]
seg_indices = ones(Int, size(token_indices2)...)

masks = Transformers.Basic.getmask(textCpoy)

sample = (tok = token_indices, segment = segment_indices)

embeddings = bert_model.embed(sample)
embeddings2 = bert_model.embed(tok=token_indices2, segment=seg_indices)


representations = bert_model.transformers(embeddings)
representations2 = bert_model.transformers(embeddings2, masks)

#Chain Bert with a linear classification layer

#
hidden_size = size(bert_model.classifier.pooler.W ,1)
labels = ("0", "1")

const clf = Chain(
    Dropout(0.1),
    Dense(hidden_size, length(labels)),
    logsoftmax
)

const bert_model_with_classifier =
    set_classifier(bert_model,
                   (
                       pooler = bert_model.classifier.pooler,
                       clf = clf
                   )
                  )

#put representation througth the classifier

a = bert_model_with_classifier.transformers(embeddings,nothing)


#try
    t = bert_model_with_classifier.transformers(embeddings, nothing)
    l = bert_model_with_classifier.classifier.clf(
            bert_model_with_classifier.classifier.pooler(
                t[:,1,:]
            )
        )

b = bert_model_with_classifier.classifier.clf(
										bert_model_with_classifier.classifier.pooler(
											a[:,1,:]
										)
									)


bert_embedding = sample |> bert_model.embed
feature_tensors = bert_embedding |> bert_model.transformers



#eval the model on data

function loss(data, label, mask=nothing)
    e = bert_model.embed(data)
    t = bert_model.transformers(e, mask)

    p = bert_model.classifier.clf(
        bert_model.classifier.pooler(
            t[:,1,:]
        )
    )

    l = Basic.logcrossentropy(label, p)
    return l, p
end

#const vocab = Vocabulary(wordpiece)

markline(sent) = ["[CLS]"; sent; "[SEP]"]

function preprocess(batch)
    sentence = markline.(wordpiece.(tokenizer.(batch[1])))
    mask = getmask(sentence)
    tok = vocab(sentence)
    segment = fill!(similar(tok), 1)

    label = onehotbatch(batch[2], labels)
    return (tok = tok, segment = segment), label, mask
end

segment = fill!(similar(tok), 1)
tok = vocab(markline.(wordpiece.(tokenizer.("Hund"))))
data = (tok,segment)

bert_model_with_classifier.embed(data)
bert_model_with_classifier.transformers(data)
bert_model_with_classifier.classifier(data)

preprocess(data)
loss(data,0)

#try 2
processed_sample = wordpiece.(tokenizer.(sample))
