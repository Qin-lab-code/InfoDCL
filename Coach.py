import numpy as np
import torch
from tqdm import tqdm
from Args import args
from diff import *
from Log import log_print
from utility import *
from collections import defaultdict



class Coach:
    def __init__(self, handler):
        self.handler = handler
        self.n_user = handler.n_user
        self.n_item = handler.n_item
        self.dim = args.hidden_factor
        log_print("USER", self.n_user, "ITEM", self.n_item)
        log_print("NUM OF INTERACTIONS", self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ["Loss", "preLoss", "prediction", "Recall", "NDCG"]
        for met in mets:
            self.metrics["Train" + met] = list()
            self.metrics["Test" + met] = list()
        setup_seed(args.random_seed)

    def makePrint(self, name, ep, reses, save):
        ret = "Epoch %d/%d, %s: " % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += "%s = %.4f, " % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + "  "
        return ret

    def run(self):
        self.prepareModel()
        log_print("Model Prepared")
        log_print("All Parameters:")
        for arg, value in vars(args).items():
            log_print(f"{arg}: {value}")

        best_epoch = 0
        best_recall_20 = 0
        best_results = ()
        trnLoader = self.handler.trnLoader
        log_print("Model Initialized")
        et = 0
        for ep in range(0, args.epoch):
            if et > 1000:
                break
            start_time = time.time()
            tstFlag = ep % args.tstEpoch == 0
            loss_dict = self.trainEpoch(trnLoader=trnLoader)
            if args.report_epoch:
                if ep % 1 == 0:
                    log_str = "Epoch {:03d}; ".format(ep)
                    for k, v in loss_dict.items():
                        log_str += "{}: {:.4f}; ".format(k, v)
                    log_str += "Time cost: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                    log_print(log_str)
            if tstFlag:
                eval_start = time.time()
                reses = self.testEpoch()
                if reses["Recall"][1] > best_recall_20:
                    best_epoch = ep
                    best_recall_20 = reses["Recall"][1]
                    best_results = reses
                    if not os.path.exists("./result"):
                        os.makedirs("./result")
                    torch.save(
                        self.model_1.state_dict(),
                        os.path.join("./result", "best_model.pt"),
                    )
                else:
                    et = et + 1
                log_print(
                    "Evalution cost: "
                    + time.strftime("%H: %M: %S", time.gmtime(time.time() - eval_start))
                )
                print_results(None, test_result=reses)
                log_print(
                    "----------------------------------------------------------------"
                )
            # save_best_result(best_results, args.data, args.name)
        print_results(test_result=best_results)

    def prepareModel(self):
        self.topN = [10, 20, 50, 100]
        self.device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )
        self.model_1 = Tenc(
            args.hidden_factor,
            args.text_dim,
            self.n_item,
            args.statesize,
            args.dropout_rate,
            args.diffuser_type,
            self.device,
        )
        self.model_2 = Tenc(
            args.hidden_factor,
            args.image_dim,
            self.n_item,
            args.statesize,
            args.dropout_rate,
            args.diffuser_type,
            self.device,
        )
        self.diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)
        self.iEmb = nn.Embedding(self.n_item, args.hidden_factor).to(self.device)
        self.uEmb = nn.Embedding(self.n_user, args.hidden_factor).to(self.device)
        self.interactions = self.load_interactions()
        self.iEmb_pt = torch.from_numpy(np.load(f"{args.data_dir}/{args.data}/iEmbeds_pt.npy")).float().to(self.device)
        self.uEmb_pt = torch.from_numpy(np.load(f"{args.data_dir}/{args.data}/uEmbeds_pt.npy")).float().to(self.device)
        self.load_add_iEmb()
        nn.init.normal_(self.iEmb.weight, std=0.1)
        nn.init.normal_(self.uEmb.weight, std=0.1)
        params_to_optimize = (
            list(self.model_1.parameters())
            + list(self.model_2.parameters())
            + list(self.iEmb.parameters())
            + list(self.uEmb.parameters())
        )

        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay
            )
        elif args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay
            )
        elif args.optimizer == "adagrad":
            self.optimizer = torch.optim.Adagrad(
                params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay
            )
        elif args.optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay
            )

        self.model_1.to(self.device)
        self.model_2.to(self.device)
        # self.iEmb=torch.tensor(self.handler.iEmb, dtype=torch.float32).to(self.device)
        # self.uEmb=torch.tensor(self.handler.uEmb, dtype=torch.float32).to(self.device)
        # self.tEmb = torch.tensor(self.handler.tEmb, dtype=torch.float32).to(self.device)
        # self.mEmb = torch.tensor(self.handler.mEmb, dtype=torch.float32).to(self.device)
        self.graph = self.handler.graph.to(self.device)

    def trainEpoch(self, trnLoader=None):
        for j, batch in enumerate(trnLoader):
            user, item = batch
            user = user.long().to(self.device)
            pos = item.long().to(self.device)
            negs = trnLoader.dataset.negSampling(pos, user)
            neg = torch.tensor(negs).long().to(self.device)
            self.optimizer.zero_grad()
            x_start = self.iEmb(pos)
            n = torch.randint(
                0, args.timesteps, (x_start.shape[0],), device=self.device
            ).long()
            add_info_1 = self.iKge[pos]
            reconloss_1, predicted_x_1 = self.diff.p_losses(
                self.model_1, x_start, add_info_1, n, loss_type="l2", flag=args.flag
            )
            add_info_2 = self.iKge[pos]
            reconloss_2, predicted_x_2 = self.diff.p_losses(
                self.model_2, x_start, add_info_2, n, loss_type="l2", flag=args.flag
            )
            user_emb0 = self.uEmb(user)
            pos_emb0 = self.iEmb(pos)
            neg_emb0 = self.iEmb(neg)
            pos_scores = torch.mul(user_emb0, pos_emb0)
            pos_scores = torch.sum(pos_scores, dim=1)
            neg_scores = torch.mul(user_emb0, neg_emb0)
            neg_scores = torch.sum(neg_scores, dim=1)
            bprloss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
            sslloss_1 = calc_ssl_loss_1(
                pos, predicted_x_1, self.iEmb.weight, tau=0.5
            )
            sslloss_2 = calc_ssl_loss_1(
                pos, predicted_x_2, self.iEmb.weight, tau=0.5
            )
            regloss = (
                (1 / 2)
                * (
                    user_emb0.norm(2).pow(2)
                    + pos_emb0.norm(2).pow(2)
                )
                / len(user)
            )
            predicted_x_loss = predicted_x_1.norm(2).pow(2) / len(item) + predicted_x_2.norm(2).pow(2) / len(item)
            loss = (
                reconloss_1 * args.recon_weight
                + reconloss_2 * args.recon_weight
                + bprloss * args.bpr_alpha
                + regloss * args.reg_alpha
                + predicted_x_loss * args.prex_weight
                + (sslloss_1 + sslloss_2) * args.ssl_weight
                + sslloss_1 * args.ssl_weight
            )
            loss.backward()
            self.optimizer.step()
        return {
            "total_loss": loss,
        }

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg, epPrecision, epMRR = None, None, None, None
        num = math.ceil(len(tstLoader.dataset) / args.tstBat)
        for usr, trnMask in tstLoader:
            usr = usr.long().to(self.device)
            trnMask_tensor = trnMask.clone().detach().to(self.device)
            tst_users_tensor = usr.clone().detach().to(self.device)
            prediction = compute_prediction(
                self.uEmb.weight,
                self.iEmb.weight,
                args.n_layers,
                self.graph,
                self.n_user,
                self.n_item,
                tst_users_tensor,
                self.model_1,
            )
            prediction = prediction * (1 - trnMask_tensor) - trnMask_tensor * 1e8
            _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
            topK = topK.cpu().detach().numpy().tolist()
            predict_items = []
            predict_items.extend(topK)
            tstLocs_np = np.array(
                self.handler.tstLoader.dataset.tstLocs, dtype=object
            )  
            target_items = tstLocs_np[usr.cpu()]  
            precision, recall, NDCG, MRR = computeTopNAccuracy(
                target_items, predict_items, self.topN
            )

            def accumulate(epMetric, metric):
                if epMetric is None:
                    return metric
                else:
                    return [epMetric[j] + metric[j] for j in range(len(metric))]

            epRecall = accumulate(epRecall, recall)
            epNdcg = accumulate(epNdcg, NDCG)
            epPrecision = accumulate(epPrecision, precision)
            epMRR = accumulate(epMRR, MRR)
        ret = dict()
        ret["Recall"] = [x / num for x in epRecall]
        ret["NDCG"] = [x / num for x in epNdcg]
        ret["Precision"] = [x / num for x in epPrecision]
        ret["MRR"] = [x / num for x in epMRR]
        return ret
    
    def gcn(self, user_emb, item_emb):
        all_emb = torch.cat([user_emb, item_emb])
        embs = [all_emb]
        for layer in range(args.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users_final, items_final = torch.split(light_out, [self.n_user, self.n_item])
        return users_final, items_final
    
    def load_interactions(self):
        npy_path = os.path.join(args.data_dir, args.data, 'train_list.npy')
        txt_path = os.path.join(args.data_dir, args.data, 'train.txt')

        if os.path.exists(npy_path):
            try:
                data = np.load(npy_path, allow_pickle=True)
                return data
            except Exception as e:
                print(f"Failed to load npy file: {e}")

        if os.path.exists(txt_path):
            data = []
            with open(txt_path, 'r') as f:
                for line in f:
                    try:
                        user, item = map(int, line.strip().split())
                        data.append((user, item))
                    except ValueError as ve:
                        print(f"Skipping invalid line: {line.strip()} — {ve}")
            return data

        print("No valid data file found.")
        return []


    def get_social_guided_item_embeddings(self):
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        for u, i in self.interactions:
            self.user_items[u].add(i)
            self.item_users[i].add(u)

        self.user_sim = torch.zeros((self.n_user, self.n_user), device=self.device)
        for u1 in tqdm(range(self.n_user), desc="compute user similarity"):
            for u2 in range(u1 + 1, self.n_user):
                inter1 = self.user_items[u1]
                inter2 = self.user_items[u2]
                if not inter1 or not inter2:
                    continue
                sim = len(inter1 & inter2) / (len(inter1 | inter2) + 1e-8)
                if sim > 0:
                    self.user_sim[u1, u2] = self.user_sim[u2, u1] = sim

        u_social_emb = self.uEmb_pt.clone()
        for u in tqdm(range(self.n_user), desc="generate user social embeddings"):
            neighbors = self.user_sim[u]
            sim_sum = neighbors.sum()
            if sim_sum > 0:
                weighted_sum = torch.matmul(neighbors, self.uEmb_pt)
                u_social_emb[u] = (self.uEmb_pt[u] + weighted_sum / sim_sum) / 2

        item_social_emb = torch.zeros((self.n_item, self.dim), device=self.device)
        for i in tqdm(range(self.n_item)):
            users = list(self.item_users[i])
            if users:
                item_social_emb[i] = u_social_emb[users].mean(dim=0)
            else:
                item_social_emb[i] = self.iEmb_pt[i]

        self.iSoc = item_social_emb
        np.save(args.data_dir + f"/{args.data}/item_social_emb.npy", item_social_emb.cpu().numpy())
        print("✅ item_social_emb.npy saved!")

    def get_item_knowledge_graph_embeddings(self, topk=20):
        self.item_sim = torch.zeros((self.n_item, self.n_item), device=self.device)
        for i in tqdm(range(self.n_item), desc="compute item-item Jaccard similarity"):
            for j in range(i + 1, self.n_item):
                users_i = self.item_users[i]
                users_j = self.item_users[j]
                if not users_i or not users_j:
                    continue
                sim = len(users_i & users_j) / (len(users_i | users_j) + 1e-8)
                if sim > 0:
                    self.item_sim[i, j] = self.item_sim[j, i] = sim

        item_kg_emb = torch.zeros((self.n_item, self.dim), device=self.device)
        for i in tqdm(range(self.n_item), desc="aggregate item neighbor embeddings"):
            sim_row = self.item_sim[i]
            topk_indices = torch.topk(sim_row, k=topk).indices
            valid = sim_row[topk_indices] > 0
            neighbors = topk_indices[valid]
            if len(neighbors) > 0:
                item_kg_emb[i] = self.iEmb_pt[neighbors].mean(dim=0)
            else:
                item_kg_emb[i] = self.iEmb_pt[i]

        self.iKge = item_kg_emb
        np.save(args.data_dir + f"/{args.data}/item_knowledge_emb.npy", item_kg_emb.cpu().numpy())
        print("✅ item_knowledge_emb.npy saved!")

    def load_add_iEmb(self):
        social_emb_path = os.path.join(args.data_dir, args.data, "item_social_emb.npy")
        if os.path.exists(social_emb_path):
            self.iSoc = torch.from_numpy(np.load(social_emb_path)).float().to(self.device)
        else:
            self.get_social_guided_item_embeddings()

        knowledge_emb_path = os.path.join(args.data_dir, args.data, "item_knowledge_emb.npy")
        if os.path.exists(knowledge_emb_path):
            self.iKge = torch.from_numpy(np.load(knowledge_emb_path)).float().to(self.device)
        else:
            self.get_item_knowledge_graph_embeddings()
