function hico_eval_wrapper(dets_file, anno_file, bbox_file)
    % Loading annotations
    anno = load(anno_file);
    bbox = load(bbox_file);

    % Loading detection results
    load(dets_file);
    assert(length(dets) == 9658);
    % Convert detection results to HICO-DET required format
    all_boxes = cell(600, 9658);
    for i = 1:length(dets)
        dets_per_img = dets{i};
        for j = 1:size(dets_per_img, 1)
            hoi_ind    = dets_per_img(j, 1) + 1; % MATLAB indices start from 1
            person_box = dets_per_img(j, 2:5);
            object_box = dets_per_img(j, 6:9);
            score      = dets_per_img(j, 10);

            if isempty(all_boxes{hoi_ind, i})
                all_boxes{hoi_ind, i} = [person_box, object_box, score];
            else
                all_boxes{hoi_ind, i} = [all_boxes{hoi_ind, i}; [person_box, object_box, score]];
            end
        end
    end

    res_def = hico_official_eval('def', all_boxes, anno, bbox);

    % Save results
    res.def_mAP_full      = res_def.mAP_full;
    res.def_mAP_rare      = res_def.mAP_rare;
    res.def_mAP_non_rare  = res_def.mAP_non_rare;

    res.def_mRec_full     = res_def.mRec_full;
    res.def_mRec_rare     = res_def.mRec_rare;
    res.def_mRec_non_rare = res_def.mRec_non_rare;

    % Print results
    fprintf("Evaluation results for HICO-DET:\n")
    fprintf("Def mAP  |   full   |   rare   |  non-rare |\n");
    fprintf("---------|:--------:|:--------:|:---------:|\n");
    fprintf("         |  %6.3f  |  %6.3f  |   %6.3f  |\n", res_def.mAP_full, res_def.mAP_rare, res_def.mAP_non_rare);
    fprintf("\n");
    fprintf("Def mRec |   full   |   rare   |  non-rare |\n");
    fprintf("--------:|:--------:|:--------:|:---------:|\n");
    fprintf("         |  %6.3f  |  %6.3f  |   %6.3f  |\n", res_def.mRec_full, res_def.mRec_rare, res_def.mRec_non_rare);

    % res_ko  = hico_official_eval('ko',  all_boxes, anno, bbox);
    % fprintf("KO mAP   |   full   |   rare   |  non-rare |\n");
    % fprintf("--------------------------------------------\n");
    % fprintf("         |  %6.3f  |  %6.3f  |   %6.3f  |\n", res_ko.mAP_full, res_ko.mAP_rare, res_ko.mAP_non_rare);
    % fprintf("--------------------------------------------\n");
    % fprintf("KO mRec  |   full   |   rare   |  non-rare |\n");
    % fprintf("--------------------------------------------\n");
    % fprintf("         |  %6.3f  |  %6.3f  |   %6.3f  |\n", res_ko.mRec_full, res_ko.mRec_rare, res_ko.mRec_non_rare);
    % fprintf("--------------------------------------------\n");

    % res.ko_mAP_full       = res_ko.mAP_full;
    % res.ko_mAP_rare       = res_ko.mAP_rare;
    % res.ko_mAP_non_rare   = res_ko.mAP_non_rare;

    % res.ko_mRec_full      = res_ko.mRec_full;
    % res.ko_mRec_rare      = res_ko.mRec_rare;
    % res.ko_mRec_non_rare  = res_ko.mRec_non_rare;

    res_name = [dets_file(1:end-4), '_res.mat'];
    save(res_name, 'res');
