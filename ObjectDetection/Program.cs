
using Microsoft.ML;
using ObjectDetection;
using ObjectDetection.DataStructures;
using ObjectDetection.YoloParser;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

const string assetsRelativePath = "./Assets";
var assetsPath = GetAbsolutePath(assetsRelativePath);
var modelFilePath = Path.Combine(assetsPath, "Models", "TinyYolo2_model.onnx");
var imagesFolder = Path.Combine(assetsPath, "Images");
var outputFolder = Path.Combine(assetsPath, "Images", "Output");

// Initialize MLContext
var mlContext = new MLContext();

try
{
    // Load Data
    var images = ImageNetData.ReadFromFile(imagesFolder);
    var imageDataView = mlContext.Data.LoadFromEnumerable(images);

    // Create instance of model scorer
    var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);

    // Use model to score data
    var probabilities = modelScorer.Score(imageDataView);

    // Post-process model output
    var parser = new YoloOutputParser();

    var boundingBoxes =
        probabilities
            .Select(probability => parser.ParseOutputs(probability))
            .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

    // Draw bounding boxes for detected objects in each of the images
    for (var i = 0; i < images.Count(); i++)
    {
        var imageFileName = images.ElementAt(i).Label;
        var detectedObjects = boundingBoxes.ElementAt(i);

        DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);

        LogDetectedObjects(imageFileName, detectedObjects);
    }
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}

Console.WriteLine("========= End of Process..Hit any Key ========");
return;

string GetAbsolutePath(string relativePath)
{
    var _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    var assemblyFolderPath = _dataRoot.Directory.FullName;

    var fullPath = Path.Combine(assemblyFolderPath, relativePath);

    return fullPath;
}

void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName,
    IList<YoloBoundingBox> filteredBoundingBoxes)
{
    using (var image = Image.Load<Rgba32>(Path.Combine(inputImageLocation, imageName)))
    {
        var originalImageHeight = image.Height;
        var originalImageWidth = image.Width;

        var font = SystemFonts.CreateFont("Arial", 24, FontStyle.Bold);

        foreach (var box in filteredBoundingBoxes)
        {
            // Get Bounding Box Dimensions
            var x = (uint)Math.Max(box.Dimensions.X, 0);
            var y = (uint)Math.Max(box.Dimensions.Y, 0);
            var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
            var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

            // Resize To Image
            x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.ImageWidth;
            y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.ImageHeight;
            width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.ImageWidth;
            height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.ImageHeight;

            // Bounding Box Text
            var text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

            // Define Text and Bounding Box options
            var boxColor = new Rgba32(box.BoxColor.R, box.BoxColor.G, box.BoxColor.B, box.BoxColor.A);
            var pen = new SolidPen(boxColor, 3.2f);
            var textColor = Rgba32.ParseHex("#FFFF00");

            // Draw bounding box and text on image
            image.Mutate(ctx =>
            {
                ctx.Draw(pen, new RectangleF(x, y, width, height));
                ctx.DrawText(new DrawingOptions { GraphicsOptions = new GraphicsOptions { Antialias = true } }, text,
                    font, textColor, new PointF(x, y - 20));
            });
        }

        if (!Directory.Exists(outputImageLocation)) Directory.CreateDirectory(outputImageLocation);

        image.Save(Path.Combine(outputImageLocation, imageName));
    }
}

void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
{
    Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

    foreach (var box in boundingBoxes) Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");

    Console.WriteLine("");
}