# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import log
from random import random


class NGram(object):

    def __init__(self, n, sents, corpus=''):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        corpus -- which corpus is being used
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        self.corpus = corpus
        sents = list(map((lambda x: ['<s>']*(n-1) + x), sents))
        sents = list(map((lambda x: x + ['</s>']), sents))

        for sent in sents:
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

    # obsolete now...
    def prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        aux_count = self.counts[tuple(tokens)]
        return aux_count / float(self.counts[tuple(prev_tokens)])

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.
        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self.counts[tokens]

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        if not prev_tokens:
            assert self.n == 1
            prev_tokens = tuple()
        # ngram condicional probs are based on relative counts
        hits = self.count((tuple(prev_tokens)+(token,)))
        sub_count = self.count(tuple(prev_tokens))

        return hits / float(sub_count)

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
        sent -- the sentence as a list of tokens.
        """

        prob = 1.0
        sent = ['<s>']*(self.n-1)+sent+['</s>']

        for i in range(self.n-1, len(sent)):
            prob *= self.cond_prob(sent[i], tuple(sent[i-self.n+1:i]))
            if not prob:
                break

        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
        sent -- the sentence as a list of tokens.
        """

        prob = 0
        sent = ['<s>']*(self.n-1)+sent+['</s>']

        for i in range(self.n-1, len(sent)):
            c_p = self.cond_prob(sent[i], tuple(sent[i-self.n+1:i]))
            # to catch a math error
            if not c_p:
                return float('-inf')
            prob += log(c_p, 2)

        return prob

    def perplexity(self, sents):
        """ Perplexity of a model.
        sents -- the test corpus as a list of sents
        """
        # total words seen
        M = 0
        for sent in sents:
            M += len(sent)
        # cross-entropy
        l = 0
        print('Computing Perplexity on {} sents...\n'.format(len(sents)))
        for sent in sents:
            l += self.sent_log_prob(sent) / M
        return pow(2, -l)

    def get_special_param(self):
        return None, None


class AddOneNGram(NGram):

    def __init__(self, n, sents, corpus=''):
        NGram.__init__(self, n, sents, corpus='')
        # way more efficient than using set union
        voc = ['</s>']
        for s in sents:
            voc += s
        self.voc = list(set(voc))
        self.corpus = corpus
        self.smoothingtechnique = 'Add One (Laplace) Smoothing'
        sents = list(map((lambda x: x + ['</s>']), sents))

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if not prev_tokens:
            assert self.n == 1
            prev_tokens = tuple()

        hits = self.count((tuple(prev_tokens)+(token,)))
        sub_count = self.count(tuple(prev_tokens))
        # heuristic
        return (hits+1) / (float(sub_count)+self.V())

    def V(self):
        """Size of the vocabulary.
        """
        return len(self.voc)


class InterpolatedNGram(AddOneNGram):

    def __init__(self, n, sents, corpus='', gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        corpus -- which corpus is being used
        """
        self.n = n
        self.smoothingtechnique = 'Interpolated (Jelinek Mercer) Smoothing'
        self.gamma = gamma
        self.addone = addone
        self.counts = counts = defaultdict(int)
        self.gamma_flag = True
        self.corpus = corpus
        # way more efficient than use set unions
        voc = ['</s>']
        for s in sents:
            voc += s
        self.voc = list(set(voc))

        if gamma is None:
            self.gamma_flag = False

        # if not gamma given
        if not self.gamma_flag:
            total_sents = len(sents)
            aux = int(total_sents * 90 / 100)
            # 90 per cent for training
            train_sents = sents[:aux]
            # 10 per cent for perplexity (held out data)
            held_out_sents = sents[-total_sents+aux:]

            train_sents = list(map((lambda x: ['<s>']*(n-1) + x), train_sents))
            train_sents = list(map((lambda x: x + ['</s>']), train_sents))

            for sent in train_sents:
                for j in range(n+1):
                    # move along the sent saving all its j-grams
                    for i in range(n-j, len(sent) - j + 1):
                        ngram = tuple(sent[i: i + j])
                        counts[ngram] += 1
            # added by hand
            counts[('</s>',)] = len(train_sents)
            # variable only for tests
            self.tocounts = counts
            # search the gamma that gives lower perplexity
            gamma_candidates = [i*50 for i in range(1, 15)]
            # xs is a list with (gamma, perplexity)
            xs = []
            sents = train_sents
            for aux_gamma in gamma_candidates:
                self.gamma = aux_gamma
                aux_perx = self.perplexity(held_out_sents)
                xs.append((aux_gamma, aux_perx))
            xs.sort(key=lambda x: x[1])
            self.gamma = xs[0][0]
            with open('old-stuff/interpolated_' + str(n) + '_parameters_'+corpus, 'a') as f:
                f.write('Order: {}\n'.format(self.n))
                f.write('Gamma: {}\n'.format(self.gamma))
                f.write('AddOne: {}\n'.format(self.addone))
                f.write('Perplexity observed: {}\n'.format(xs[0][1]))
                f.write('-------------------------------\n')
            f.close()

        else:
            sents = list(map((lambda x: ['<s>']*(n-1) + x), sents))
            sents = list(map((lambda x: x + ['</s>']), sents))

            for sent in sents:
                # counts now holds all k-grams for 0 < k < n + 1
                for j in range(n+1):
                    # move along the sent saving all its j-grams
                    for i in range(n-j, len(sent) - j + 1):
                        ngram = tuple(sent[i: i + j])
                        counts[ngram] += 1
            # added by hand
            counts[('</s>',)] = len(sents)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        addone = self.addone
        n = self.n
        gamma = self.gamma

        if not prev_tokens:
            prev_tokens = []
            assert len(prev_tokens) == n - 1

        lambdas = []
        for i in range(0, n-1):
            # 1 - sum(previous lambdas)
            aux_lambda = 1 - sum(lambdas)
            # counts for numerator
            counts_top = self.count(tuple(prev_tokens[i:n-1]))
            # counts plus gamma (denominator)
            counts_w_gamma = self.count(tuple(prev_tokens[i:n-1])) + gamma
            # save the computed i-th lambda
            lambdas.append(aux_lambda * (counts_top / counts_w_gamma))
        # last lambda, by hand
        lambdas.append(1-sum(lambdas))

        # Maximum likelihood probs
        ML_probs = dict()
        for i in range(0, n):
            hits = self.count((tuple(prev_tokens[i:])+(token,)))
            sub_count = self.count(tuple(prev_tokens[i:]))
            result = 0
            if addone and not len(prev_tokens[i:]):
                result = (hits+1) / (float(sub_count) + len(self.voc))
            else:
                if sub_count:
                    result = hits / float(sub_count)
            # the (i+1)-th element in ML_probs holds the q_ML value
            # for a (n-i)-gram
            ML_probs[i+1] = result

        prob = 0
        # ML_probs dict starts in 1
        for j in range(0, n):
            prob += ML_probs[j+1] * lambdas[j]
        return prob

    def get_special_param(self):
        return "Gamma", self.gamma


class BackOffNGram(NGram):

    def __init__(self, n, sents, corpus='', beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        corpus -- which corpus is being used
        """
        self.n = n
        self.beta = beta
        self.corpus = corpus
        self.beta_flag = True
        self.addone = addone
        self.smoothingtechnique = 'Back Off (Katz) with Discounting Smoothing'
        self.counts = counts = defaultdict(int)
        self.A_set = defaultdict(set)
        voc = ['</s>']
        for s in sents:
            voc += s
        self.voc = set(voc)
        if beta is None:
            self.beta_flag = False

        # if no beta given, we compute it
        if not self.beta_flag:
            total_sents = len(sents)
            aux = int(total_sents * 90 / 100)
            # 90 per cent por training
            train_sents = sents[:aux]
            # 10 per cent for perplexity (held out data)
            held_out_sents = sents[-total_sents+aux:]

            train_sents = list(map((lambda x: ['<s>']*(n-1) + x), train_sents))
            train_sents = list(map((lambda x: x + ['</s>']), train_sents))
            for sent in train_sents:
                for j in range(n+1):
                    for i in range(n-j, len(sent) - j + 1):
                        ngram = tuple(sent[i: i + j])
                        counts[ngram] += 1
                        # for efficiency, we save the A set as a dict of sets
                        if j:
                            self.A_set[ngram[:-1]].add(ngram[-1])
            for i in range(1, n):
                counts[('<s>',)*i] += len(train_sents)
            counts[('</s>',)] = len(train_sents)

            self.tocounts = counts
            # search for the beta that gives lower perplexity
            beta_candidates = [i*0.1 for i in range(1, 10)]
            # xs is a list with (beta, perplexity)
            xs = []
            self.sents = train_sents
            for aux_beta in beta_candidates:
                self.beta = aux_beta
                aux_perx = self.perplexity(held_out_sents)
                xs.append((aux_beta, aux_perx))
            xs.sort(key=lambda x: x[1])
            self.beta = xs[0][0]
            with open('old-stuff/backoff_'+str(n)+'_parameters_'+corpus, 'a') as f:
                f.write('Order: {}\n'.format(self.n))
                f.write('Beta: {}\n'.format(self.beta))
                f.write('AddOne: {}\n'.format(self.addone))
                f.write('Perplexity observed: {}\n'.format(xs[0][1]))
                f.write('-------------------------------\n')
            f.close()
        else:
            sents = list(map((lambda x: x + ['</s>']), sents))
            sents = list(map((lambda x: ['<s>']*(n-1) + x), sents))

            for sent in sents:
                for j in range(n+1):
                    for i in range(n-j, len(sent) - j + 1):
                        ngram = tuple(sent[i: i + j])
                        counts[ngram] += 1
                        # for efficiency, we save the A set as a dict of sets
                        if j:
                            self.A_set[ngram[:-1]].add(ngram[-1])
            for i in range(1, n):
                counts[('<s>',)*i] += len(sents)
            counts[('</s>',)] = len(sents)

    # c*() counts
    def count_star(self, tokens):
        """
        Discounting counts for counts > 0
        """
        return self.counts[tokens] - self.beta

    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
        tokens -- the k-gram tuple.
        """

        if not tokens:
            tokens = []
        return self.A_set[tuple(tokens)]

    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
        tokens -- the k-gram tuple.
        """
        if not tokens:
            tokens = tuple()

        A_set = self.A(tokens)
        result = 1
        # heuristic, way more efficient
        if len(A_set):
            result = self.beta * len(A_set) / self.count(tuple(tokens))
        return result

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        addone = self.addone

        # unigram case
        if not prev_tokens:
            if addone:
                result = (self.count((token,))+1) / (self.V() + self.count(()))
            else:
                result = self.count((token,)) / self.count(())
        else:
            A_set = self.A(prev_tokens)
            # check if discounting can be applied
            if token in A_set:
                result = self.count_star(tuple(prev_tokens) + (token,)) /\
                    self.count(tuple(prev_tokens))
            else:
                # recursive call
                q_D = self.cond_prob(token, prev_tokens[1:])
                denom_factor = self.denom(prev_tokens)
                if denom_factor:
                    alpha = self.alpha(prev_tokens)
                    result = alpha * q_D / denom_factor
                else:
                    result = 0
        return result

    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
        tokens -- the k-gram tuple.
        """

        sum = 0
        A_set = self.A(tokens)
        # heuristic
        for elem in A_set:
            sum += self.cond_prob(elem, tokens[1:])
        return 1 - sum

    def V(self):
        """Size of the vocabulary.
        """
        return len(self.voc)

    def get_special_param(self):
        return "Beta", self.beta


class KneserNeyBaseNGram(NGram):
    def __init__(self, sents, n, corpus='', D=None):
        """
        sents -- list of sents
        n -- order of the model
        D -- Discount value
        """

        self.n = n
        self.D = D
        self.corpus = corpus
        self.smoothingtechnique = 'Kneser Ney Smoothing'
        # N1+(·w_<i+1>)
        self._N_dot_tokens_dict = N_dot_tokens = defaultdict(set)
        # N1+(w^<n-1> ·)
        self._N_tokens_dot_dict = N_tokens_dot = defaultdict(set)
        # N1+(· w^<i-1>_<i-n+1> ·)
        self._N_dot_tokens_dot_dict = N_dot_tokens_dot = defaultdict(set)
        self.counts = counts = defaultdict(int)
        vocabulary = []

        if D is None:
            total_sents = len(sents)
            k = int(total_sents*9/10)
            training_sents = sents[:k]
            held_out_sents = sents[k:]
            training_sents = list(map(lambda x: ['<s>']*(n-1) + x + ['</s>'], training_sents))
            for sent in training_sents:
                for j in range(n+1):
                    for i in range(n-j, len(sent) - j + 1):
                        ngram = tuple(sent[i: i + j])
                        counts[ngram] += 1
                        if ngram:
                            if len(ngram) == 1:
                                vocabulary.append(ngram[0])
                            else:
                                right_token, left_token, right_kgram, left_kgram, middle_kgram =\
                                    ngram[-1:], ngram[:1], ngram[1:], ngram[:-1], ngram[1:-1]
                                N_dot_tokens[right_kgram].add(left_token)
                                N_tokens_dot[left_kgram].add(right_token)
                                if middle_kgram:
                                    N_dot_tokens_dot[middle_kgram].add(right_token)
                                    N_dot_tokens_dot[middle_kgram].add(left_token)
            if n - 1:
                counts[('<s>',)*(n-1)] = len(sents)
            self.vocab = set(vocabulary)
            aux = 0
            for w in self.vocab:
                aux += len(self._N_dot_tokens_dict[(w,)])
            self._N_dot_dot_attr = aux
            D_candidates = [i*0.12 for i in range(1, 9)]
            xs = []
            for D in D_candidates:
                self.D = D
                aux_perplexity = self.perplexity(held_out_sents)
                xs.append((D, aux_perplexity))
            xs.sort(key=lambda x: x[1])
            self.D = xs[0][0]
            with open('old-stuff/kneserney_' + str(n) + '_parameters_'+corpus, 'a') as f:
                f.write('Order: {}\n'.format(self.n))
                f.write('D: {}\n'.format(self.D))
                f.write('Perplexity observed: {}\n'.format(xs[0][1]))
                f.write('-------------------------------\n')
            f.close()

        # discount value D provided
        else:
            sents = list(map(lambda x: ['<s>']*(n-1) + x + ['</s>'], sents))
            for sent in sents:
                for j in range(n+1):
                    # all k-grams for 0 <= k <= n
                    for i in range(n-j, len(sent) - j + 1):
                        ngram = tuple(sent[i: i + j])
                        counts[ngram] += 1
                        if ngram:
                            if len(ngram) == 1:
                                vocabulary.append(ngram[0])
                            else:
                                # e.g., ngram = (1,2,3,4,5,6,7,8)
                                # right_token = (8,)
                                # left_token = (1,)
                                # right_kgram = (2,3,4,5,6,7,8)
                                # left_kgram = (1,2,3,4,5,6,7)
                                # middle_kgram = (2,3,4,5,6,7)
                                right_token, left_token, right_kgram, left_kgram, middle_kgram =\
                                    ngram[-1:], ngram[:1], ngram[1:], ngram[:-1], ngram[1:-1]
                                N_dot_tokens[right_kgram].add(left_token)
                                N_tokens_dot[left_kgram].add(right_token)
                                if middle_kgram:
                                    N_dot_tokens_dot[middle_kgram].add(right_token)
                                    N_dot_tokens_dot[middle_kgram].add(left_token)
            if n-1:
                counts[('<s>',)*(n-1)] = len(sents)
            self.vocab = set(vocabulary)

            aux = 0
            for w in self.vocab:
                aux += len(self._N_dot_tokens_dict[(w,)])
            self._N_dot_dot_attr = aux

            xs = [k for k, v in counts.items() if v == 1 and n == len(k)]
            ys = [k for k, v in counts.items() if v == 2 and n == len(k)]
            n1 = len(xs)
            n2 = len(ys)
            self.D = n1 / (n1 + 2 * n2)

    def V(self):
        """
        returns the size of the vocabulary
        """
        return len(self.vocab)

    def N_dot_dot(self):
        """
        Returns the sum of N_dot_token(w) for all w in the vocabulary
        """
        return self._N_dot_dot_attr

    def N_tokens_dot(self, tokens):
        """
        Returns a set of words in which count(prev_tokens+word) > 0
        i.e., the different ngrams it completes

        tokens -- a tuple of strings
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_tokens_dot_dict[tokens]

    def N_dot_tokens(self, tokens):
        """
        Returns a set of ngrams it completes

        tokens -- a tuple of strings
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_dot_tokens_dict[tokens]

    def N_dot_tokens_dot(self, tokens):
        """
        Returns a set of ngrams it completes

        tokens -- a tuple of strings
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_dot_tokens_dot_dict[tokens]

    def get_special_param(self):
        return "D", self.D


# From https://west.uni-koblenz.de/sites/default/files/BachelorArbeit_MartinKoerner.pdf

class KneserNeyNGram(KneserNeyBaseNGram):
    def __init__(self, sents, n, corpus='', D=None):
        super(KneserNeyNGram, self).__init__(sents=sents, corpus=corpus, n=n, D=D)

    def cond_prob(self, token, prev_tokens=tuple()):
        n = self.n
        # two cases:
        # 1) n == 1
        # 2) n > 1:
           # 2.1) k == 1
           # 2.2) 1 < k < n
           # 2.3) k == n

        # case 1)
        # heuristic addone
        if not prev_tokens and n == 1:
            return (self.count((token,))+1) / (self.count(()) + self.V())

        # case 2.1)
        # lowest ngram
        if not prev_tokens and n > 1:
            aux1 = len(self.N_dot_tokens((token,)))
            aux2 = self.N_dot_dot()
            # addone smoothing
            return (aux1 + 1) / (aux2 + self.V())

        # highest ngram
        if len(prev_tokens) == n-1:
            c = self.count(prev_tokens) + 1
            t1 = max(self.count(prev_tokens+(token,)) - self.D, 0) / c
            # addone smoothing
            t2 = self.D * max(len(self.N_tokens_dot(prev_tokens)), 1) / c
            t3 = self.cond_prob(token, prev_tokens[1:])
            return t1 + t2 * t3
        # lower ngram
        else:
            # addone smoothing
            aux = max(len(self.N_dot_tokens_dot(prev_tokens)), 1)
            t1 = max(len(self.N_dot_tokens(prev_tokens+(token,))) - self.D, 0) / aux
            t2 = self.D * max(len(self.N_tokens_dot(prev_tokens)), 1) / aux
            t3 = self.cond_prob(token, prev_tokens[1:])
            return t1 + t2 * t3


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.n = model.n
        self.probs = probs = dict()
        self.sorted_probs = dict()
        # pre, list of grams with length n-1 (of a n-gram model)
        pre = [elem for elem in model.counts.keys() if not len(elem) == self.n]
        # suf, list of grams with length n (of a n-gram model)
        suf = [elem for elem in model.counts.keys() if len(elem) == self.n]

        for elem in suf:
            prfx = elem[:-1]
            sfx = elem[-1]
            # if prfx already in dict, we add the new sufix and its
            # probability and update the dict
            if prfx in probs:
                aux = probs[prfx]
                # probs values are dicts with (token, cond_prob of token)
                probs[prfx] = {sfx: model.cond_prob(sfx, prfx)}
                probs[prfx].update(aux)
            else:
                probs[prfx] = {sfx: model.cond_prob(sfx, prfx)}
        # order the dict by its values with higher probability
        # so we can use the inverse transform method
        pre = [i[0] for i in probs.items()]

        sp = [list(probs[x].items()) for x in pre]

        self.sorted_probs = {pre[i]:
                                 sorted(sp[i], key=lambda x: (-x[1], x[0]))
                             for i in range(len(sp))}

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self.n
        sent = ('<s>',)*(n-1)
        if n == 1:
            sent = ()
        # generate until STOP symbol comes up
        while '</s>' not in sent:
            sent += (self.generate_token(sent[-n+1:]),)
        return sent[n-1:-1]

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        if n == 1:
            prev_tokens = tuple()
        p = random()
        res = ''
        choices = self.sorted_probs[prev_tokens]

        # applying the inverse transform method
        acc = choices[0][1]
        for i in range(0, len(choices)):
            if p < acc:
                res = choices[i][0]
                break
            else:
                acc += choices[i][1]
        return res
