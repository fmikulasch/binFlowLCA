package.path = package.path .. ";" .. "/home/neuralnetlab/workspace/OpenPV/parameterWrapper/?.lua"
local pv = require "PVModule"

folder = "/home/neuralnetlab/workspace/Projects/BinFlowLCA/"
leftInputPath       = {};
leftInputPath[0]    = folder .. "input/inputImagesLeft0.txt";
leftInputPath[1]    = folder .. "input/inputImagesLeft1.txt";
leftInputPath[2]    = folder .. "input/inputImagesLeft2.txt";
leftInputPath[3]    = folder .. "input/inputImagesLeft3.txt";
rightInputPath       = {};
rightInputPath[0]    = folder .. "input/inputImagesRight0.txt";
rightInputPath[1]    = folder .. "input/inputImagesRight1.txt";
rightInputPath[2]    = folder .. "input/inputImagesRight2.txt";
rightInputPath[3]    = folder .. "input/inputImagesRight3.txt";
inputPath = {
	Left = leftInputPath;
	Right =	rightInputPath
}


local nbatch           = 32;    --Batch size of learning
local nxScale          = 1; 
local nyScale          = 1;
local iSize            = 32;
local nxSize           = iSize; 
local nySize           = iSize;
local npsize	       = iSize;  --One kernel for the whole image
local stride           = 1;

local displayPeriod    = 400;   --Number of timesteps to find sparse approximation
local numEpochs        = 4;     --Number of times to run through dataset
local numImages        = 6158; --Total number of images in dataset
local stopTime         = math.ceil((numImages  * numEpochs) / nbatch) * displayPeriod;

local basisVectorFile  = nil;   --nil for initial weights, otherwise, specifies the weights file to load.
local plasticityFlag   = true;  --Determines if we are learning weights or holding them constant

local numBasisVectors  = 1 * 32 * 32 * 8;  --overcompleteness x (stride X) x (Stride Y) * (# color channels) * (2 if rectified) 
local momentumTau      = 100;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
local dWMax            = 6;    --The learning rate
local VThresh          = .02; --.025;  -- .005; --The threshold, or lambda, of the network
local AMin             = 0;
local AMax             = infinity;
local AShift           = .02;--0.15; --VThresh;  --This being equal to VThresh is a soft threshold
local VWidth           = 0;
local timeConstantTau = 100; --The integration tau for sparse approximation
local weightInit = math.sqrt((1/iSize)*(1/iSize)*(1/3));

local outputPath          = folder .. "output/";
local checkpointWriteDir  = folder .. "output/checkpoints/";
local checkpointPeriod = displayPeriod * 250; -- How often to write checkpoints. Was display*100
local writeStep        = (stopTime - 1) / 4; 
local initialWriteTime = 0;--writeStep;

-- Base table variable to store
local pvParameters = {
column = {
groupType = "HyPerCol";
    startTime                           = 0;
    dt                                  = 1;
    dtAdaptFlag                         = true;
    writeTimeScaleFieldnames            = true;
    useAdaptMethodExp1stOrder           = true;
    dtAdaptController                   = "V1EnergyProbe";
    dtAdaptTriggerLayerName             = "LeftImageAxis0";
    dtAdaptTriggerOffset                = 0;
    dtScaleMax                          = 0.01; -- .1;
    dtScaleMin                          = 0.001;
    dtChangeMax                         = 0.001;
    dtChangeMin                         = 0.00001; --0;
    dtMinToleratedTimeScale             = 0.00001;
    stopTime                            = stopTime;
    progressInterval                    = checkpointPeriod;
    writeProgressToErr                  = true;
    verifyWrites                        = false;
    outputPath                          = outputPath;
    printParamsFilename                 = "pv.params";
    randomSeed                          = 1234567890;
    nx                                  = nxSize;
    ny                                  = nySize;
    nbatch                              = nbatch;
    filenamesContainLayerNames          = 2;
    filenamesContainConnectionNames     = 2;
    initializeFromCheckpointDir         = "";
    defaultInitializeFromCheckpointFlag = false;
    checkpointWrite                     = true;
    checkpointWriteDir                  = checkpointWriteDir;
    checkpointWriteTriggerMode          = "step";
    checkpointWriteStepInterval         = checkpointPeriod;
    deleteOlderCheckpoints              = false;
    suppressNonplasticCheckpoints       = false;
    checkpointIndexWidth                = -1;
    writeTimescales                     = true;
    errorOnNotANumber                   = false;
};

V1 = {
groupType = "HyPerLCALayer";
    nxScale                             = nxScale/nxSize;
    nyScale                             = nyScale/nySize;
    nf                                  = numBasisVectors;
    phase                               = 2;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    InitVType                           = "ConstantV";
    --minV                                = -1;
    --maxV                                = 0.02;
    valueV                              = VThresh; --
    triggerLayerName                    = nil;
    writeStep                           = writeStep;
    initialWriteTime                    = initialWriteTime;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
    VThresh                             = VThresh;
    AMin                                = 0;
    AMax                                = infinity;
    AShift                              = AShift;
    VWidth                              = VWidth;
    clearGSynInterval                   = 0;
    timeConstantTau                     = timeConstanTau;
    selfInteract                        = true;
};

V1EnergyProbe = {
groupType = "ColumnEnergyProbe";
    message                             = nil;
    textOutputFlag                      = true;
    probeOutputFile                     = "V1EnergyProbe.txt";
    triggerLayerName                    = nil;
    energyProbe                         = nil;
};

V1L1NormEnergyProbe = {
groupType = "L1NormProbe";
    targetLayer                         = "V1";
    message                             = nil;
    textOutputFlag                      = true;
    probeOutputFile                     = "V1L1NormEnergyProbe.txt";
--     triggerLayerName                    = nil;
    energyProbe                         = "V1EnergyProbe";
    coefficient                         = 0.025;
    maskLayerName                       = nil;
};

}

side = {'Left', 'Right'}

for s=1,2,1 -- sides 
do
for i=0,3,1 -- axis
do

------------------ Layers -----------------	

	pvParameters[side[s] .. "ImageAxis" .. i] = {
		groupType = "Movie";
		    nxScale                             = nxScale;
		    nyScale                             = nyScale;
		    nf                                  = 1;
		    phase                               = 0;
		    mirrorBCflag                        = true;
		    writeStep                           = writeStep;
		    initialWriteTime 		        = initialWriteTime;
		    sparseLayer                         = false;
		    updateGpu                           = false;
		    dataType                            = nil;
		    inputPath                           = inputPath[side[s]][i];
		    offsetAnchor                        = "br";
		    offsetX                             = 0;
		    offsetY                             = 0;
		    writeImages                         = 0;
		    autoResizeFlag                      = false;
		    inverseFlag                         = false;
		    normalizeLuminanceFlag              = true;
		    normalizeStdDev                     = true;
		    jitterFlag                          = 0;
		    useImageBCflag                      = false;
		    padValue                            = 0;
		    displayPeriod                       = displayPeriod;
		    echoFramePathnameFlag               = true;
		    batchMethod                         = "byImage";
		    writeFrameToTimestamp               = true;
		    flipOnTimescaleError                = true;
		    resetToStartOnLoop                  = false;
		};

	pvParameters[side[s] .. "ErrorAxis" .. i] = {
		groupType = "ANNNormalizedErrorLayer";
		    nxScale                             = nxScale;
		    nyScale                             = nyScale;
		    nf                                  = 1;
		    phase                               = 1;
		    mirrorBCflag                        = false;
		    valueBC                             = 0;
		    InitVType                           = "ZeroV";
		    triggerLayerName                    = nil;
		    writeStep                           = writeStep;
		    initialWriteTime                    = initialWriteTime;
		    sparseLayer                         = false;
		    updateGpu                           = false;
		    dataType                            = nil;
		    VThresh                             = -infinity; --0;
		    AMin                                = -infinity;
		    AMax                                = infinity;
		    AShift                              = 0;
		    clearGSynInterval                   = 0;
-- 		    errScale                            = 1;
		    useMask                             = false;
		};

	pvParameters[side[s] .. "ReconAxis" .. i] = {
		groupType = "ANNLayer";
		    nxScale                             = nxScale;
		    nyScale                             = nyScale;
		    nf                                  = 1;
		    phase                               = 3;
		    mirrorBCflag                        = false;
		    valueBC                             = 0;
		    InitVType                           = "ZeroV";
		    triggerLayerName                    = nil; -- side[s] .. "ImageAxis" .. i;
-- 		    triggerOffset                       = 1;
-- 		    triggerBehavior                     = "updateOnlyOnTrigger";
		    writeStep                           = writeStep;
		    initialWriteTime                    = initialWriteTime;
		    sparseLayer                         = false;
		    updateGpu                           = false;
		    dataType                            = nil;
		    VThresh                             = -infinity;
		    AMin                                = -infinity;
		    AMax                                = infinity;
		    AShift                              = 0;
		    VWidth                              = 0;
		    clearGSynInterval                   = 0;
		};


------------------ Connections -----------------	

	pvParameters[side[s] .. "ImageTo" .. side[s] .. "ErrorAxis" .. i] = {
		groupType = "RescaleConn";
		    preLayerName                        = side[s] .. "ImageAxis" .. i;
		    postLayerName                       = side[s] .. "ErrorAxis" .. i;
		    channelCode                         = 0;
		    delay                               = {0.000000};
		    scale                               = weightInit;
		};

	pvParameters[side[s] .. "ErrorToV1Axis" .. i] = {
		groupType = "TransposeConn";
		    preLayerName                        = side[s] .. "ErrorAxis" .. i;
		    postLayerName                       = "V1";
		    channelCode                         = 0;
		    delay                               = {0.000000};
		    convertRateToSpikeCount             = false;
		    receiveGpu                          = true;
		    updateGSynFromPostPerspective       = true;
		    pvpatchAccumulateType               = "convolve";
		    writeStep                           = -1;
		    writeCompressedCheckpoints          = false;
		    selfFlag                            = false;
		    gpuGroupIdx                         = -1;
		    --weightSparsity                      = 0;
		    originalConnName                    = "V1To" .. side[s] .. "ErrorAxis" .. i;
		};

	pvParameters["V1To" .. side[s] .."ErrorAxis" .. i] = {
		groupType = "MomentumConn";
		    preLayerName                        = "V1";
		    postLayerName                       = side[s] .. "ErrorAxis" .. i;
		    channelCode                         = -1; 
		    delay                               = {0.000000};
		    numAxonalArbors                     = 1;
		    plasticityFlag                      = true;
		    convertRateToSpikeCount             = false;
		    receiveGpu                          = false;
		    sharedWeights                       = true;
		    weightInitType                      = "UniformRandomWeight";
		    initWeightsFile                     = nil;
		    wMinInit                            = -1;
		    wMaxInit                            = 1;
		    sparseFraction                      = 0.9;
		    triggerLayerName                    = side[s] .. "ImageAxis" .. i;
		    triggerOffset                       = 0; --1;
		    updateGSynFromPostPerspective       = false;
		    pvpatchAccumulateType               = "convolve";
		    writeStep                           = writeStep;
		    initialWriteTime                    = initialWriteTime;
		    writeCompressedCheckpoints          = false;
		    selfFlag                            = false;
-- 		    combine_dW_with_W_flag              = false;
		    nxp                                 = npsize;
		    nyp                                 = npsize;
-- 		    nfp                                 = 1;
		    shrinkPatches                       = false;
		    normalizeMethod                     = "normalizeL2";
		    strength                            = 1;
		    normalizeArborsIndividually         = false;
		    normalizeOnInitialize               = true;
		    normalizeOnWeightUpdate             = true;
		    rMinX                               = 0;
		    rMinY                               = 0;
		    nonnegativeConstraintFlag           = false;
		    normalize_cutoff                    = 0;
		    normalizeFromPostPerspective        = false;
		    minL2NormTolerated                  = 0;
		    dWMax                               = dWMax;
		    keepKernelsSynchronized             = true;
		    -- normalizeDw                         = true;
		    useMask                             = false;
		    momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
		    momentumMethod                      = "viscosity";
		    momentumDecay 			= 0;
		    -- weightSparsity                      = 0;
		};

	pvParameters["V1To" .. side[s] .. "ReconAxis" .. i] = {
		groupType = "CloneConn";
		    preLayerName                        = "V1";
		    postLayerName                       = side[s] .. "ReconAxis" .. i;
		    channelCode                         = 0;
		    delay                               = {0.000000};
		    convertRateToSpikeCount             = false;
		    receiveGpu                          = false;
		    updateGSynFromPostPerspective       = false;
		    pvpatchAccumulateType               = "convolve";
		    selfFlag                            = false;
		    originalConnName                    = "V1To" .. side[s] .. "ErrorAxis" .. i;
		};
	
	pvParameters[side[s] .. "ReconToErrorAxis" .. i] = {
	     	groupType = "IdentConn";
		    preLayerName                        = side[s] .. "ReconAxis" .. i;
		    postLayerName                       = side[s] .. "ErrorAxis" .. i;
		    channelCode                         = 1;
		    delay                               = {0.000000};
		    initWeightsFile                     = nil;
		};

	pvParameters[side[s] .. "ErrorL2NormEnergyProbeAxis" .. i] = {
		groupType = "L2NormProbe";
		    targetLayer                         = side[s] .. "ErrorAxis" .. i;
		    message                             = nil;
		    textOutputFlag                      = true;
		    probeOutputFile                     = side[s] .. "ErrorL2NormEnergyProbeAxis" .. i .. ".txt";
		    triggerLayerName                    = nil;
		    energyProbe                         = "V1EnergyProbe";
		    coefficient                         = 0.125; -- 1/8 images
		    maskLayerName                       = nil;
		    exponent                            = 2;
		};

end
end



-- Print out PetaVision approved parameter file to the console
pv.printConsole(pvParameters)
