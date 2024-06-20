Creation

    subword-nmt learn-bpe -s 4000 < total_train.txt > 4000_codes
    subword-nmt apply-bpe -c 4000_codes < train.uz > 4000_train.uz
    subword-nmt apply-bpe -c 4000_codes < val.uz > 4000_val.uz
    subword-nmt apply-bpe -c 4000_codes < test.uz > 4000_test.uz

    subword-nmt apply-bpe -c 4000_codes < train.en > 4000_train.en
    subword-nmt apply-bpe -c 4000_codes < val.en > 4000_val.en
    subword-nmt apply-bpe -c 4000_codes < test.en > 4000_test.en




















# # # # # #

    Provo a RIPESCARE i dati NON "BPE-izzati" facendo
        sed e rimpiazzando '@@ ' con solo spazio...

            hu@hu-X580VD:/media/hu/DATA/PASSAGGIO_WINDOWS/Proj_LowRes_Giulia/lowres_workspace/bpe_data/qed_kk_en$ cat 4000_test.en | sed s'/@@ //'g > test.en
            hu@hu-X580VD:/media/hu/DATA/PASSAGGIO_WINDOWS/Proj_LowRes_Giulia/lowres_workspace/bpe_data/qed_kk_en$ cat 4000_train.en | sed s'/@@ //'g > train.en
            hu@hu-X580VD:/media/hu/DATA/PASSAGGIO_WINDOWS/Proj_LowRes_Giulia/lowres_workspace/bpe_data/qed_kk_en$ cat 4000_train.kk | sed s'/@@ //'g > train.kk
            hu@hu-X580VD:/media/hu/DATA/PASSAGGIO_WINDOWS/Proj_LowRes_Giulia/lowres_workspace/bpe_data/qed_kk_en$ cat 4000_test.kk | sed s'/@@ //'g > test.kk
            hu@hu-X580VD:/media/hu/DATA/PASSAGGIO_WINDOWS/Proj_LowRes_Giulia/lowres_workspace/bpe_data/qed_kk_en$ cat 4000_val.kk | sed s'/@@ //'g > val.kk
            hu@hu-X580VD:/media/hu/DATA/PASSAGGIO_WINDOWS/Proj_LowRes_Giulia/lowres_workspace/bpe_data/qed_kk_en$ cat 4000_val.en | sed s'/@@ //'g > val.en

# # # # # #

    Provo a fare 16000_BPE ... nella speranza
    che aumentino i frammenti di BPE...

        cat train.uz train.en > total_train.txt

        subword-nmt learn-bpe -s 16000 < total_train.txt > more_bpe/uz_16000_codes
        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < train.uz > more_bpe/16000_train.uz
        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < val.uz > more_bpe/16000_val.uz
        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < test.uz > more_bpe/16000_test.uz

        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < train.en > more_bpe/16000_train.en
        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < val.en > more_bpe/16000_val.en
        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < test.en > more_bpe/16000_test.en

        37 dice no pairs has length >= 2...
        metto 16.000


        subword-nmt learn-bpe -s 16000 < total_train.txt > more_bpe/uz_16000_codes
        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < train.uz > more_bpe/16000_train.uz
        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < val.uz > more_bpe/16000_val.uz
        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < test.uz > more_bpe/16000_test.uz

        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < train.en > more_bpe/16000_train.en
        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < val.en > more_bpe/16000_val.en
        subword-nmt apply-bpe -c more_bpe/uz_16000_codes < test.en > more_bpe/16000_test.en
