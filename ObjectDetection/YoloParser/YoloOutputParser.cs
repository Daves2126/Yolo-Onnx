namespace ObjectDetection.YoloParser;

using System.Drawing;

internal class YoloOutputParser
{
    private const int RowCount = 13;
    private const int ColCount = 13;
    private const int ChannelCount = 125;
    private const int BoxesPerCell = 5;
    private const int BoxInfoFeatureCount = 5;
    private const int ClassCount = 20;
    private const float CellWidth = 32;
    private const float CellHeight = 32;

    private static readonly Color[] ClassColors =
    [
        Color.Khaki,
        Color.Fuchsia,
        Color.Silver,
        Color.RoyalBlue,
        Color.Green,
        Color.DarkOrange,
        Color.Purple,
        Color.Gold,
        Color.Red,
        Color.Aquamarine,
        Color.Lime,
        Color.AliceBlue,
        Color.Sienna,
        Color.Orchid,
        Color.Tan,
        Color.LightPink,
        Color.Yellow,
        Color.HotPink,
        Color.OliveDrab,
        Color.SandyBrown,
        Color.DarkTurquoise
    ];

    private readonly float[] _anchors =
    [
        1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
    ];

    private const int ChannelStride = RowCount * ColCount;

    private readonly string[] _labels =
    [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ];

    private static float Sigmoid(float value)
    {
        var k = (float)Math.Exp(value);
        return k / (1.0f + k);
    }

    private static float[] Softmax(float[] values)
    {
        var maxVal = values.Max();
        var exp = values.Select(v => Math.Exp(v - maxVal));
        var sumExp = exp.Sum();

        return exp.Select(v => (float)(v / sumExp)).ToArray();
    }

    private static int GetOffset(int x, int y, int channel)
    {
        return channel * ChannelStride + y * ColCount + x;
    }

    private static BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
    {
        return new BoundingBoxDimensions
        {
            X = modelOutput[GetOffset(x, y, channel)],
            Y = modelOutput[GetOffset(x, y, channel + 1)],
            Width = modelOutput[GetOffset(x, y, channel + 2)],
            Height = modelOutput[GetOffset(x, y, channel + 3)]
        };
    }

    private static float GetConfidence(float[] modelOutput, int x, int y, int channel)
    {
        return Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]);
    }

    private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
    {
        return new CellDimensions
        {
            X = (x + Sigmoid(boxDimensions.X)) * CellWidth,
            Y = (y + Sigmoid(boxDimensions.Y)) * CellHeight,
            Width = (float)Math.Exp(boxDimensions.Width) * CellWidth * _anchors[box * 2],
            Height = (float)Math.Exp(boxDimensions.Height) * CellHeight * _anchors[box * 2 + 1]
        };
    }

    public static float[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
    {
        var predictedClasses = new float[ClassCount];
        var predictedClassOffset = channel + BoxInfoFeatureCount;
        for (var predictedClass = 0; predictedClass < ClassCount; predictedClass++)
            predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
        return Softmax(predictedClasses);
    }

    private static ValueTuple<int, float> GetTopResult(float[] predictedClasses)
    {
        return predictedClasses
            .Select((predictedClass, index) => (Index: index, Value: predictedClass))
            .OrderByDescending(result => result.Value)
            .First();
    }

    private static float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
    {
        var areaA = boundingBoxA.Width * boundingBoxA.Height;

        if (areaA <= 0)
            return 0;

        var areaB = boundingBoxB.Width * boundingBoxB.Height;

        if (areaB <= 0)
            return 0;

        var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
        var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
        var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
        var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

        var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    public IList<YoloBoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .3F)
    {
        var boxes = new List<YoloBoundingBox>();

        for (var row = 0; row < RowCount; row++)
        for (var column = 0; column < ColCount; column++)
        for (var box = 0; box < BoxesPerCell; box++)
        {
            var channel = box * (ClassCount + BoxInfoFeatureCount);

            var boundingBoxDimensions = ExtractBoundingBoxDimensions(yoloModelOutputs, row, column, channel);

            var confidence = GetConfidence(yoloModelOutputs, row, column, channel);

            var mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxDimensions);

            if (confidence < threshold)
                continue;

            var predictedClasses = ExtractClasses(yoloModelOutputs, row, column, channel);

            var (topResultIndex, topResultScore) = GetTopResult(predictedClasses);
            var topScore = topResultScore * confidence;

            if (topScore < threshold)
                continue;

            boxes.Add(new YoloBoundingBox
            {
                Dimensions = new BoundingBoxDimensions
                {
                    X = mappedBoundingBox.X - mappedBoundingBox.Width / 2,
                    Y = mappedBoundingBox.Y - mappedBoundingBox.Height / 2,
                    Width = mappedBoundingBox.Width,
                    Height = mappedBoundingBox.Height
                },
                Confidence = topScore,
                Label = _labels[topResultIndex],
                BoxColor = ClassColors[topResultIndex]
            });
        }

        return boxes;
    }

    public IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
    {
        var activeCount = boxes.Count;
        var isActiveBoxes = new bool[boxes.Count];

        for (var i = 0; i < isActiveBoxes.Length; i++)
            isActiveBoxes[i] = true;

        var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
            .OrderByDescending(b => b.Box.Confidence)
            .ToList();

        var results = new List<YoloBoundingBox>();

        for (var i = 0; i < boxes.Count; i++)
            if (isActiveBoxes[i])
            {
                var boxA = sortedBoxes[i].Box;
                results.Add(boxA);

                if (results.Count >= limit)
                    break;

                for (var j = i + 1; j < boxes.Count; j++)
                    if (isActiveBoxes[j])
                    {
                        var boxB = sortedBoxes[j].Box;

                        if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                        {
                            isActiveBoxes[j] = false;
                            activeCount--;

                            if (activeCount <= 0)
                                break;
                        }
                    }

                if (activeCount <= 0)
                    break;
            }

        return results;
    }

    private class CellDimensions : DimensionsBase
    {
    }
}