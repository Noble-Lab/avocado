# models.py
# Contact: Jacob Schreiber <jmschr@cs.washington.edu>
#          William Noble   <wnoble@uw.edu>

"""
Avocado is deep tensor factorization model for learning a latent representation
of the human epigenome. This file has functions for building a deep tensor
factorization model.
"""

def build_model(n_celltypes, n_celltype_factors, n_assays, n_assay_factors,
	n_genomic_positions, n_25bp_factors, n_250bp_factors, n_5kbp_factors,
	n_layers, n_nodes):
	"""This function builds a multi-scale deep tensor factorization model."""

	celltype_input = Input(shape=(1,), name="celltype")
	celltype_embedding = Embedding(n_celltypes, n_celltype_factors, input_length=1)(celltype_input)
	celltype = Flatten()(celltype_embedding)

	assay_input = Input(shape=(1,), name="assay")
	assay_embedding = Embedding(n_assays, n_assay_factors, input_length=1)
	assay = Flatten()(assay_embedding(assay_input))

	genome_25bp_input = Input(shape=(1,), name="genome_25bp")
	genome_25bp_embedding = Embedding(n_genomic_positions, n_25bp_factors, input_length=1)
	genome_25bp = Flatten()(genome_25bp_embedding(genome_25bp_input))

	genome_250bp_input = Input(shape=(1,), name="genome_250bp")
	genome_250bp_embedding = Embedding((n_genomic_positions / 10) + 1, n_250bp_factors, input_length=1)
	genome_250bp = Flatten()(genome_250bp_embedding(genome_250bp_input))

	genome_5kbp_input = Input(shape=(1,), name="genome_5kbp")
	genome_5kbp_embedding = Embedding((n_genomic_positions / 200) + 1, n_5kbp_factors, input_length=1)
	genome_5kbp = Flatten()(genome_5kbp_embedding(genome_5kbp_input))

	layers = [celltype, assay, genome_25bp, genome_250bp, genome_5kbp]
	inputs = (celltype_input, assay_input, genome_25bp_input, genome_250bp_input, genome_5kbp_input)

	x = concatenate(layers)
	for i in range(n_layers):
		x = Dense(n_nodes, activation='relu')(x)

	y = Dense(1)(x)

	model = Model(inputs=inputs, outputs=y)
	model.compile(optimizer='adam', loss='mse', metrics=['mse'])
	return model

class Avocado(object):
	def __init__(self, celltypes, assays, n_celltypes=127, n_celltype_factors=32, 
		n_assays=24, n_assay_factors=256, n_genomic_positions=1126468, 
		n_25bp_factors=25, n_250bp_factors=40, n_5kbp_factors=45, n_layers=2,
		n_nodes=2048):
		self.celltypes = celltypes
		self.assays = assays

		self.n_celltypes = len(celltypes)
		self.n_assays = len(assays)

		self.model = build_model(n_celltypes=n_celltypes,
								 n_celltype_factors=n_celltype_factors,
								 n_assays=n_assays,
								 n_assay_factors=n_assay_factors,
								 n_genomic_positions=n_genomic_positions,
								 n_25bp_factors=n_25bp_factors,
								 n_250bp_factors=n_250bp_factors,
								 n_5kbp_factors=n_5kbp_factors,
								 n_layers=n_layers,
								 n_nodes=n_nodes)




