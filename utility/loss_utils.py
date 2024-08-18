import torch as t
import torch.nn.functional as F

def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
	pos_preds = (anc_embeds * pos_embeds).sum(-1)
	neg_preds = (anc_embeds * neg_embeds).sum(-1)
	return t.sum(F.softplus(neg_preds - pos_preds))


def reg_pick_embeds(embeds_list):
	reg_loss = 0
	for embeds in embeds_list:
		reg_loss += embeds.square().sum()
	return reg_loss


def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
	normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
	normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
	normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
	nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
	deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
	cl_loss = (nume_term + deno_term).sum()
	return cl_loss


def cal_infonce_loss_spec_nodes(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1 + 1e-8, p=2)
	embeds2 = F.normalize(embeds2 + 1e-8, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
	return -t.log(nume / deno).mean()


def cal_sce_loss(x, y, alpha):
	x = F.normalize(x, p=2, dim=-1)
	y = F.normalize(y, p=2, dim=-1)
	loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
	loss = loss.mean()
	return loss


def cal_rank_loss(stu_anc_emb, stu_pos_emb, stu_neg_emb, tea_anc_emb, tea_pos_emb, tea_neg_emb):
	stu_pos_score = (stu_anc_emb * stu_pos_emb).sum(dim=-1)
	stu_neg_score = (stu_anc_emb * stu_neg_emb).sum(dim=-1)
	stu_r_score = F.sigmoid(stu_pos_score - stu_neg_score)

	tea_pos_score = (tea_anc_emb * tea_pos_emb).sum(dim=-1)
	tea_neg_score = (tea_anc_emb * tea_neg_emb).sum(dim=-1)
	tea_r_score = F.sigmoid(tea_pos_score - tea_neg_score)

	rank_loss = -(tea_r_score * t.log(stu_r_score + 1e-8) + (1 - tea_r_score) * t.log(1 - stu_r_score + 1e-8)).mean()

	return rank_loss


def reg_params(model):
	reg_loss = 0
	for W in model.parameters():
		reg_loss += W.norm(2).square()
	return reg_loss