function run_dparsf_pipeline(raw_dir, out_dir)
% RUN_DPARSF_PIPELINE  BrainCL-style preprocessing for rs-fMRI.
%
%   Steps:
%     1. Slice-timing correction
%     2. Realignment (24-parameter Friston model)
%     3. Nuisance regression (WM, CSF, 24 motion parameters)
%     4. Normalization to MNI 3x3x3
%     5. Smoothing with FWHM = 4 mm
%     6. Band-pass filtering 0.01-0.10 Hz
%     7. Scrubbing: FD_Jenkinson > 0.5 mm flagged; sessions with mean FD > 0.3 dropped

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    cfg = struct();
    cfg.DataProcessDir   = raw_dir;
    cfg.OutputDir        = out_dir;
    cfg.StartingDirName  = 'FunRaw';
    cfg.IsNeedConvertFunDCM2IMG   = 0;
    cfg.IsRemoveFirstTimePoints   = 1;
    cfg.RemoveFirstTimePoints     = 10;
    cfg.IsSliceTiming             = 1;
    cfg.IsRealign                 = 1;
    cfg.IsCovremove               = 1;
    cfg.Covremove.Timing          = 'BeforeNormalize';
    cfg.Covremove.WM              = 1;
    cfg.Covremove.CSF             = 1;
    cfg.Covremove.HeadMotion      = 24;    % Friston 24-parameter
    cfg.Covremove.PolynomialTrend = 1;
    cfg.IsNormalize               = 2;      % EPI template
    cfg.Normalize.VoxSize         = [3 3 3];
    cfg.IsSmooth                  = 1;
    cfg.Smooth.FWHM               = [4 4 4];
    cfg.IsFilter                  = 1;
    cfg.Filter.ABandLow           = 0.01;
    cfg.Filter.ABandHigh          = 0.10;
    cfg.IsScrubbing               = 1;
    cfg.Scrubbing.FDThreshold     = 0.5;
    cfg.Scrubbing.MeanFDThreshold = 0.3;
    cfg.IsExtractROISignals       = 1;
    cfg.ROISignalsAtlases         = {'CC200', 'AAL3'};

    DPARSFA_run(cfg);

    fprintf('[run_dparsf_pipeline] Completed: %s\n', out_dir);
end
