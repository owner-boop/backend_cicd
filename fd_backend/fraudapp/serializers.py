from rest_framework import serializers

class JSONResponseSerializer(serializers.Serializer):
    json_data = serializers.JSONField()

class PieRecordsSerializer(serializers.Serializer):
    pie_labels = serializers.ListField(
        child=serializers.CharField()
    )
    pie_values = serializers.ListField(
        child=serializers.IntegerField()
    )

class ChartDataSerializer(serializers.Serializer):
    labels = serializers.ListField(
        child=serializers.CharField()
    )
    values = serializers.ListField(
        child=serializers.IntegerField()
    )

class LineChartSerializer(serializers.Serializer):
    labels = serializers.ListField(
        child=serializers.CharField()
    )
    values = serializers.ListField(
        child=serializers.FloatField()
    )

class HistogramSerializer(serializers.Serializer):
    bins = serializers.ListField(
        child=serializers.CharField()
    )
    values = serializers.ListField(
        child=serializers.IntegerField()
    )

class HeatMapProceedSerializer(serializers.Serializer):
    bins = serializers.ListField(
        child=serializers.CharField()
    )
    values = serializers.ListField(
        child=serializers.IntegerField()
    )

class BoxPlotDataSerializer(serializers.Serializer):
    InitialApprovalAmount = serializers.FloatField()
    ForgivenessAmount = serializers.FloatField()
    PROCEED_Per_Job = serializers.FloatField()

class BoxPlotSerializer(serializers.Serializer):
    box_plot_data = BoxPlotDataSerializer()

class ScatterPlotSerializer(serializers.Serializer):
    x = serializers.ListField(
        child=serializers.FloatField()
    )
    y = serializers.ListField(
        child=serializers.FloatField()
    )



class BarChartRecordSerializer(serializers.Serializer):
    pie_records = PieRecordsSerializer()
    loan_status = ChartDataSerializer()
    borrower_state = ChartDataSerializer()
    business_type = ChartDataSerializer()
    race = ChartDataSerializer()
    ethnicity = ChartDataSerializer()
    gender = ChartDataSerializer()
    veteran = ChartDataSerializer()
    line_chart_approval = LineChartSerializer()
    line_chart_forgiveness = LineChartSerializer()
    histogram_initial = HistogramSerializer()
    histogram_forgiveness = HistogramSerializer()
    histogram_proceed = HistogramSerializer()
    HeatMapProceed = HeatMapProceedSerializer()
 
class barser(serializers.Serializer):
    p = serializers.CharField()

class FraudPredictionSerializer(serializers.Serializer):
    LoanNumber = serializers.CharField()
    DateApproved = serializers.CharField()
    SBAOfficeCode = serializers.CharField()
    ProcessingMethod = serializers.CharField()
    BorrowerName = serializers.CharField()
    BorrowerAddress = serializers.CharField()
    BorrowerCity = serializers.CharField()
    BorrowerState = serializers.CharField()
    BorrowerZip = serializers.CharField()
    LoanStatusDate = serializers.CharField()
    LoanStatus = serializers.CharField()
    Term = serializers.CharField()
    SBAGuarantyPercentage = serializers.CharField()
    InitialApprovalAmount = serializers.CharField()
    CurrentApprovalAmount = serializers.CharField()
    UndisbursedAmount = serializers.CharField()
    FranchiseName = serializers.CharField()
    ServicingLenderLocationID = serializers.CharField()
    ServicingLenderName = serializers.CharField()
    ServicingLenderAddress = serializers.CharField()
    ServicingLenderCity = serializers.CharField()
    ServicingLenderState = serializers.CharField()
    ServicingLenderZip = serializers.CharField()
    RuralUrbanIndicator = serializers.CharField()
    HubzoneIndicator = serializers.CharField()
    LMIIndicator = serializers.CharField()
    BusinessAgeDescription = serializers.CharField()
    ProjectCity = serializers.CharField()
    ProjectCountyName = serializers.CharField()
    ProjectState = serializers.CharField()
    ProjectZip = serializers.CharField()
    CD = serializers.CharField()
    JobsReported = serializers.CharField()
    NAICSCode = serializers.CharField()
    Race = serializers.CharField()
    Ethnicity = serializers.CharField()
    UTILITIES_PROCEED = serializers.CharField()
    PAYROLL_PROCEED = serializers.CharField()
    MORTGAGE_INTEREST_PROCEED = serializers.CharField()
    RENT_PROCEED = serializers.CharField()
    REFINANCE_EIDL_PROCEED = serializers.CharField()
    HEALTH_CARE_PROCEED = serializers.CharField()
    DEBT_INTEREST_PROCEED = serializers.CharField()
    BusinessType = serializers.CharField()
    OriginatingLenderLocationID = serializers.CharField()
    OriginatingLender = serializers.CharField()
    OriginatingLenderCity = serializers.CharField()
    OriginatingLenderState = serializers.CharField()
    Gender = serializers.CharField()
    Veteran = serializers.CharField()
    NonProfit = serializers.CharField()
    ForgivenessAmount = serializers.CharField()
    ForgivenessDate = serializers.CharField()
    ApprovalDiff = serializers.CharField()
    NotForgivenAmount = serializers.CharField()
    ForgivenPercentage = serializers.CharField()
    TOTAL_PROCEED = serializers.CharField()
    PROCEED_Diff = serializers.CharField()
    UTILITIES_PROCEED_pct = serializers.CharField()
    PAYROLL_PROCEED_pct = serializers.CharField()
    MORTGAGE_INTEREST_PROCEED_pct = serializers.CharField()
    RENT_PROCEED_pct = serializers.CharField()
    REFINANCE_EIDL_PROCEED_pct = serializers.CharField()
    HEALTH_CARE_PROCEED_pct = serializers.CharField()
    DEBT_INTEREST_PROCEED_pct = serializers.CharField()
    PROCEED_Per_Job = serializers.CharField()
