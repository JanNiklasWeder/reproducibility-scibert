using DataDeps


register(DataDep("FastText en",
    """
    Dataset: FastText Word Embeddings for English.
    Author: Bojanowski et. al. (Facebook)
    License: CC-SA 3.0
    Website: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

    300 dimentional FastText word embeddings, trained on Wikipedia
    Citation: P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information

    Notice: this file is over 6.2GB
    """,
    "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec",
    "ba5420ac217fb34f15f58ded0d911a4370dfb1f3341fa7511a49ae74c87de282"
));

register(DataDep("WordNet 3.0",
    """
    Dataset: WordNet 3.0
    Website: https://wordnet.princeton.edu/wordnet

    George A. Miller (1995). WordNet: A Lexical Database for English.
    Communications of the ACM Vol. 38, No. 11: 39-41.

    Christiane Fellbaum (1998, ed.) WordNet: An Electronic Lexical Database. Cambridge, MA: MIT Press.

    License:
    WordNet Release 3.0 This software and database is being provided to you, the LICENSEE, by Princeton University under the following license. By obtaining, using and/or copying this software and database, you agree that you have read, understood, and will comply with these terms and conditions.: Permission to use, copy, modify and distribute this software and database and its documentation for any purpose and without fee or royalty is hereby granted, provided that you agree to comply with the following copyright notice and statements, including the disclaimer, and that the same appear on ALL copies of the software, database and documentation, including modifications that you make for internal use or for distribution. WordNet 3.0 Copyright 2006 by Princeton University. All rights reserved. THIS SOFTWARE AND DATABASE IS PROVIDED "AS IS" AND PRINCETON UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT NOT LIMITATION, PRINCETON UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES OF MERCHANT- ABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF THE LICENSED SOFTWARE, DATABASE OR DOCUMENTATION WILL NOT INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS. The name of Princeton University or Princeton may not be used in advertising or publicity pertaining to distribution of the software and/or database. Title to copyright in this software, database and any associated documentation shall at all times remain with Princeton University and LICENSEE agrees to preserve same.
    """,
    "http://wordnetcode.princeton.edu/3.0/WNdb-3.0.tar.gz",
    "658b1ba191f5f98c2e9bae3e25c186013158f30ef779f191d2a44e5d25046dc8";
    post_fetch_method = unpack
));

register(DataDep("SciBert Chemprot",
    """
    Dataset: Chemprot
    Website

    Weiter coole infos
    """
    ,
    ["https://github.com/allenai/scibert/blob/06793f77d7278898159ed50da30d173cdc8fdea9/data/text_classification/chemprot/dev.txt",
    "https://github.com/allenai/scibert/blob/06793f77d7278898159ed50da30d173cdc8fdea9/data/text_classification/chemprot/test.txt",
    "https://github.com/allenai/scibert/blob/06793f77d7278898159ed50da30d173cdc8fdea9/data/text_classification/chemprot/train.txt"]
))

datadep"SciBert Chemprot/dev.txt" |> print



using DataDeps, DataDepsGenerators, ZipFile
generate(GitHub(), "https://github.com/fivethirtyeight/data/tree/master/avengers") |> print
generate(GitHub(), "https://github.com/allenai/scibert/tree/master/data") |> print


eval(Meta.parse(generate(GitHub(), "https://github.com/allenai/scibert/tree/master/data/")))
eval(Meta.parse(generate(UCI(), "https://archive.ics.uci.edu/ml/datasets/Air+quality", "UCI Air")))

unz

function cool(file)
    unzip(file,exdir=file)
    mv(file + "scibert-master/data/",file)
    rm(scibert-master)
end

register(DataDep("SciBert",
    """
    Dataset: Chemprot
    Website

    Weiter coole infos
    """
    ,
    "https://codeload.github.com/allenai/scibert/zip/master",
    post_fetch_method = file ->(unpack(file),
                        #replace(file, ".zip" => "") ,
                        print(file,"\n"),
                        print("$(SubString(file,1,findlast(==('.'),file).-1))/data/\n"),
                        print(SubString.(file,1,findlast.(==('/'),file)), "\n"),
                        mv("$(SubString.(file,1,findlast.(==('.'),file).-1))/data/","$(SubString.(file,1,findlast.(==('/'),file)))/data" ))
))

file = "scibert.zip"
replace(file, ".zip" => "")
mv("$file/data","./data")

readdir(datadep"SciBert", join=true)
a= load(datadep"SciBert"; escapechar='"')
a= load(datadep"538/avengers.csv"; escapechar='"')
