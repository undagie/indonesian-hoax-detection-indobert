# Master experiment report

- Total runs found: **140**
- Unique experiment tags: **28**
- Total result artifacts: **1739**

## Top-5 by F1-macro (mean across seeds)

- **imb_combined**: F1=0.7240 (+/-0.0118, n_seeds=5) | Acc=0.8392 | ROC-AUC=0.8083
- **cmp_transformer_imb-combined**: F1=0.7240 (+/-0.0118, n_seeds=5) | Acc=0.8392 | ROC-AUC=0.8083
- **imb_class_weight**: F1=0.7169 (+/-0.0249, n_seeds=5) | Acc=0.8395 | ROC-AUC=0.7969
- **aug_contextual**: F1=0.7164 (+/-0.0311, n_seeds=5) | Acc=0.8364 | ROC-AUC=0.7986
- **cmp_transformer_aug-contextual**: F1=0.7164 (+/-0.0311, n_seeds=5) | Acc=0.8364 | ROC-AUC=0.7986

## Where to find things

- Per-run raw: `results/runs/<tag>/<leaf>/`
- Aggregated tables: `results/tables/master_aggregated.{csv,md,tex}`
- Per-run table: `results/tables/master_per_run.csv`
- Headline JSON: `results/tables/headline.json`
- All-models comparison plot: `results/figures/report/compare_all_models.png`
- Inventory of every artifact: `results/tables/inventory.csv`

## Full aggregated table

tag|n_seeds|f1_macro|f1_macro_std|accuracy|roc_auc|model_name|aug_method|imbalance

imb_combined|5|0.7239637137987651|0.011759268157594793|0.8392434988179669|0.8082911278989711|indolem/indobert-base-uncased|none|combined

cmp_transformer_imb-combined|5|0.7239637137987651|0.011759268157594793|0.8392434988179669|0.8082911278989711|indolem/indobert-base-uncased|none|combined

imb_class_weight|5|0.7168546519195839|0.024882108601397827|0.8394799054373523|0.796873496873497|indolem/indobert-base-uncased|none|class_weight

aug_contextual|5|0.71638824961109|0.03108331719251952|0.8364066193853429|0.7985768044591575|indolem/indobert-base-uncased|contextual|none

cmp_transformer_aug-contextual|5|0.71638824961109|0.03108331719251952|0.8364066193853429|0.7985768044591575|indolem/indobert-base-uncased|contextual|none

imb_focal_loss|5|0.7161961744150512|0.031225069653083776|0.8354609929078014|0.811106395420121|indolem/indobert-base-uncased|none|focal_loss

imb_weighted_sampler|5|0.7152669839572499|0.03512704276954623|0.8385342789598109|0.8007837478425713|indolem/indobert-base-uncased|none|weighted_sampler

cmp_transformer_vanilla|5|0.712394940862897|0.01333471276113406|0.8449172576832152|0.7935819445623368|indolem/indobert-base-uncased|none|none

imb_none|5|0.712394940862897|0.01333471276113406|0.8449172576832152|0.7935819445623368|indolem/indobert-base-uncased|none|none

aug_none|5|0.712394940862897|0.01333471276113406|0.8449172576832152|0.7935819445623368|indolem/indobert-base-uncased|none|none

cmp_transformer_proposed|5|0.7116431047860939|0.027842698079674738|0.8364066193853427|0.7971988795518207|indolem/indobert-base-uncased|contextual|combined

imb_ldam_loss|5|0.709074403643602|0.01822291622239302|0.8312056737588653|0.7958596233106036|indolem/indobert-base-uncased|none|ldam_loss

imb_random_oversample|5|0.7078861358595742|0.020860238467069525|0.841371158392435|0.7870874949306322|indolem/indobert-base-uncased|none|random_oversample

imb_cb_loss|5|0.7061165623715816|0.02456668509589876|0.8286052009456266|0.7942977864546492|indolem/indobert-base-uncased|none|cb_loss

aug_synonym|5|0.7057799663428993|0.03961834091578994|0.833096926713948|0.796288751190712|indolem/indobert-base-uncased|synonym|none

aug_delete|5|0.7033023964615965|0.028831935272495328|0.8319148936170212|0.8016363447735996|indolem/indobert-base-uncased|delete|none

aug_insert|5|0.6923762262871704|0.015111059131771042|0.8401891252955084|0.7802544586858312|indolem/indobert-base-uncased|insert|none

aug_back_translation|5|0.6898057295132384|0.018840650499480985|0.8219858156028369|0.7875166228107405|indolem/indobert-base-uncased|back_translation|none

cmp_baseline_tfidf__logreg|5|0.6877668969903843|0.013454744937584646|0.8118203309692671|0.7835912816304973|logreg|none|class_weight

baseline_tfidf__logreg|5|0.6877668969903843|0.013454744937584646|0.8118203309692671|0.7835912816304973|logreg|none|class_weight

aug_swap|5|0.6787947358275142|0.029436750721507917|0.83451536643026|0.7809316319120241|indolem/indobert-base-uncased|swap|none

imb_random_undersample|5|0.6786190390511951|0.02631451347237498|0.7728132387706855|0.7904865649963689|indolem/indobert-base-uncased|none|random_undersample

baseline_tfidf__svm|5|0.6707719498721234|0.010654579331861863|0.8210401891252955|0.7660451385941582|svm|none|class_weight

cmp_baseline_tfidf__svm|5|0.6707719498721234|0.010654579331861863|0.8210401891252955|0.7660451385941582|svm|none|class_weight

cmp_baseline_tfidf__xgboost|5|0.669501719817368|0.007271504037173258|0.8477541371158391|0.7462137717039677|xgboost|none|class_weight

baseline_tfidf__xgboost|5|0.669501719817368|0.007271504037173258|0.8477541371158391|0.7462137717039677|xgboost|none|class_weight

baseline_tfidf__rf|5|0.5896824811017455|0.023015297157317163|0.8364066193853429|0.7623951937677427|rf|none|class_weight

cmp_baseline_tfidf__rf|5|0.5896824811017455|0.023015297157317163|0.8364066193853429|0.7623951937677427|rf|none|class_weight
