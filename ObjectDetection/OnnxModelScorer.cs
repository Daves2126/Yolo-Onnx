namespace ObjectDetection;

using Microsoft.ML;
using Microsoft.ML.Data;
using DataStructures;

internal class OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
{
    private ITransformer LoadModel(string modelPath)
    {
        Console.WriteLine("Read model");
        Console.WriteLine($"Model location: {modelPath}");
        Console.WriteLine(
            $"Default parameters: image size=({ImageNetSettings.ImageWidth},{ImageNetSettings.ImageHeight})");

        // Create IDataView from empty list to obtain input data schema
        var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

        // Define scoring pipeline
        var pipeline = mlContext.Transforms.LoadImages("image", "", nameof(ImageNetData.ImagePath))
            .Append(mlContext.Transforms.ResizeImages("image", ImageNetSettings.ImageWidth,
                ImageNetSettings.ImageHeight, "image"))
            .Append(mlContext.Transforms.ExtractPixels("image"))
            .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelPath,
                outputColumnNames: [TinyYoloModelSettings.ModelOutput],
                inputColumnNames: [TinyYoloModelSettings.ModelInput]));

        // Fit scoring pipeline
        var model = pipeline.Fit(data);

        return model;
    }

    private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
    {
        Console.WriteLine($"Images location: {imagesFolder}");
        Console.WriteLine("");
        Console.WriteLine("=====Identify the objects in the images=====");
        Console.WriteLine("");

        var scoredData = model.Transform(testData);

        var probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);

        return probabilities;
    }

    public IEnumerable<float[]> Score(IDataView data)
    {
        var model = LoadModel(modelLocation);

        return PredictDataUsingModel(data, model);
    }

    public struct ImageNetSettings
    {
        public const int ImageHeight = 416;
        public const int ImageWidth = 416;
    }

    private struct TinyYoloModelSettings
    {
        public const string ModelInput = "image";
        public const string ModelOutput = "grid";
    }
}