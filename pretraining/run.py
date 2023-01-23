import logging
import os, csv
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    BertConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers import DataCollatorWithPadding
from transformers.trainer_utils import is_main_process
from tqdm import tqdm

from dataset import OurBertPretrainedDataset, OurCollator
from model import Our_pretrain
import func

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from config import (
    ModelArguments,
    DataTrainingArguments
) 

def load_geohash(args):
    dict_geohash = {}
    with open(args.geohash_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            hashcode = row[0]
            dict_geohash.setdefault(hashcode, len(dict_geohash))
    return dict_geohash

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args = TrainingArguments(
        output_dir='./output-wo-poi',
        do_train=1,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        dataloader_num_workers=4,
        logging_dir='./logs',
        save_steps=10000,
    )
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    
    print(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )


    # Set the verbosity to info of the Transformers logger (on main process only):
    
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)
    

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # data_files = {}
    # if data_args.train_file is not None:
    #     data_files["train"] = data_args.train_file
    # if data_args.validation_file is not None:
    #     data_files["validation"] = data_args.validation_file


    # config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config_decoder = BertConfig(num_hidden_layers=2)
        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        #logger.warning("You are instantiating a new config instance from scratch.")

    # tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        print(len(tokenizer))
        geo_codes = load_geohash(data_args)
        geo_special_tokens_dict = ['['+gcd+']' for gcd in geo_codes]
        tokenizer.add_tokens(geo_special_tokens_dict)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    

    # model
    if model_args.model_name_or_path:
        # initalize
        model = Our_pretrain.from_pretrained(
            model_args.model_name_or_path,
            data_args, 
            tokenizer, 
            config_decoder,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.build_network()

        #for name, param in model.named_parameters():
        #    print(name, param.size())
    else:
        logger.info("Training new model from scratch")
        model = Our_pretrain.from_config(config)

    

    # dataset
    poi_path = '/nfs/volume-93-2/meilang/myprojects/pretrain_poi_user_behavior/dataset/1_processed_data/1_poi_need_attr.csv'
    pretrain_query_task_path = '/nfs/volume-93-2/meilang/myprojects/pretrain_poi_user_behavior/dataset/1_processed_data/6_userlog_pretrain_query_task.csv'
    path_poi_task_subgraph = '/nfs/volume-93-2/meilang/myprojects/pretrain_poi_user_behavior/dataset/1_processed_data/8_neigbor2p_userlog_sample_subgraph.dict'

    dict_poi_set = func.load_data2dict(poi_path, 1, 0, [0, 1, 2, 3])
    print("finish loading poi data", len(dict_poi_set))

    pretrain_query_data = func.load_data2list(pretrain_query_task_path, 1, [0, 1, 2, 3])
    print("finish loading pretrain query data", len(pretrain_query_data))

    pretrain_poi_data = []
    dict_end_2p_edges = func.load_obj(path_poi_task_subgraph) # dict_end_2p_edges
    for end_poid in tqdm(dict_end_2p_edges):
        node_index = {}
        index_node = {}
        edges_index = {}
        for sepair in dict_end_2p_edges[end_poid]:
            startpoid, endpoid = sepair
            
            node_index.setdefault(startpoid, len(node_index))
            node_index.setdefault(endpoid, len(node_index))
            
            index_node.setdefault(node_index[startpoid], startpoid)
            index_node.setdefault(node_index[endpoid], endpoid)
            
            edges_index.setdefault((node_index[startpoid], node_index[endpoid]), 0)


        nodes = [index_node[kkk] for kkk in range(len(index_node))]
        edges = [[], []]
        for se in edges_index:
            edges[0].append(se[0])
            edges[1].append(se[1])
        
        if len(nodes) - 1 != max([kkk for kkk in range(len(index_node))]) or \
            max(edges[0]) > len(nodes) - 1 or \
            max(edges[1]) > len(nodes) - 1:
        
            print([kkk for kkk in range(len(index_node))], '\n', edges)
        assert len(nodes) - 1 == max([kkk for kkk in range(len(index_node))])
        assert max(edges[0]) <= len(nodes) - 1
        assert max(edges[1]) <= len(nodes) - 1
        
        pretrain_poi_data.append([nodes, edges, end_poid])
    print("finish loading pretrain poi data", len(pretrain_poi_data))

    stats_node_len = [len(item[0]) for item in pretrain_poi_data]
    stats_edge_len = [len(item[1][0]) for item in pretrain_poi_data]

    print('max, min, ave, node_len', max(stats_node_len), min(stats_node_len), sum(stats_node_len) / len(stats_node_len))
    print('max, min, ave, edge_len', max(stats_edge_len), min(stats_edge_len), sum(stats_edge_len) / len(stats_edge_len))


    # datasets
    print("start getting dataset....................")
    if training_args.do_train:
        train_dataset = OurBertPretrainedDataset(
            data_args, tokenizer, 
            [dict_poi_set, pretrain_query_data, pretrain_poi_data],
        )
    else:
        train_dataset = None
    print('getting dataset succcess................')

    # collator
    data_collator = OurCollator(tokenizer, data_args, [dict_poi_set, pretrain_query_data, pretrain_poi_data])
    # tokenizer, data_args, tasks
    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
        

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        #trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
    
if __name__ == "__main__":
    # main
    main()

