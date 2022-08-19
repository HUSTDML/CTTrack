from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.data.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
from lib.train.data.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data.util import dataset, opencv_loader, processing, LTRLoader, pil_loader
import lib.train.data.util.transforms as tfm


def names2datasets(name_list: list, settings, image_loader):
	assert isinstance(name_list, list)
	datasets = []
	for name in name_list:
		assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17", "VID",
		                "TRACKINGNET"]
		if name == "LASOT":
			if settings.use_lmdb:
				print("Building lasot dataset from lmdb")
				datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
			else:
				datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
		if name == "GOT10K_vottrain":
			if settings.use_lmdb:
				print("Building got10k from lmdb")
				datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
			else:
				datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
		if name == "GOT10K_train_full":
			if settings.use_lmdb:
				print("Building got10k_train_full from lmdb")
				datasets.append(
					Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
			else:
				datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
		if name == "GOT10K_votval":
			if settings.use_lmdb:
				print("Building got10k from lmdb")
				datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
			else:
				datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
		if name == "COCO17":
			if settings.use_lmdb:
				print("Building COCO2017 from lmdb")
				datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
			else:
				datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
		if name == "VID":
			if settings.use_lmdb:
				print("Building VID from lmdb")
				datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
			else:
				datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
		if name == "TRACKINGNET":
			if settings.use_lmdb:
				print("Building TrackingNet from lmdb")
				datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
			else:
				datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
	return datasets


def build_seq_dataloaders(cfg, settings):
	# Data transform
	# Data transform
	transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
	                                tfm.RandomHorizontalFlip(probability=0.5)
	                                )
	# transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0))

	transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
	                                tfm.RandomHorizontalFlip_Norm(probability=0.5),
	                                tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

	search_transform = tfm.Transform(tfm.ToTensor(),
	                                 tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

	transform_val = tfm.Transform(tfm.ToTensor(),
	                              tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

	# The tracking pairs processing module
	data_processing_train = processing.SequenceProcessing(search_area_factor=settings.search_area_factor,
	                                                      output_sz=settings.output_sz,
	                                                      center_jitter_factor=settings.center_jitter_factor,
	                                                      scale_jitter_factor=settings.scale_jitter_factor,
	                                                      transform=transform_train,
	                                                      search_transform=search_transform,
	                                                      joint_transform=transform_joint,
	                                                      settings=settings)

	data_processing_val = processing.SequenceProcessing(search_area_factor=settings.search_area_factor,
	                                                    output_sz=settings.output_sz,
	                                                    center_jitter_factor=settings.center_jitter_factor,
	                                                    scale_jitter_factor=settings.scale_jitter_factor,
	                                                    transform=transform_val,
	                                                    search_transform=search_transform,
	                                                    joint_transform=transform_joint,
	                                                    settings=settings)

	train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False) or getattr(cfg.MODEL.SCOREHEAD, "EXIST", False)

	# Train sampler and loader
	dataset_train = dataset.SequenceDataset(
		datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, pil_loader),
		p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
		samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
		num_search_frames=settings.num_search,
		num_template_frames=settings.num_template,
		num_sequence_frames=settings.num_sequence,
		processing=data_processing_train,
		train_cls=train_cls)

	train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
	shuffle = False if settings.local_rank != -1 else True
	loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
	                         num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

	# Validation samplers and loaders
	dataset_val = dataset.SequenceDataset(
		datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, pil_loader),
		p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
		samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
		num_search_frames=settings.num_search,
		num_template_frames=settings.num_template,
		num_sequence_frames=settings.num_sequence,
		processing=data_processing_val,
		train_cls=train_cls)

	val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
	loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
	                       num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
	                       epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

	return loader_train, loader_val
