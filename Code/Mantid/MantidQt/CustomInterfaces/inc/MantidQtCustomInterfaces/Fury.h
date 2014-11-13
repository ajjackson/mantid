#ifndef MANTIDQTCUSTOMINTERFACESIDA_FURY_H_
#define MANTIDQTCUSTOMINTERFACESIDA_FURY_H_

#include "MantidQtCustomInterfaces/IDATab.h"

namespace MantidQt
{
namespace CustomInterfaces
{
namespace IDA
{
  class Fury : public IDATab
  {
    Q_OBJECT

  public:
    Fury(QWidget * parent = 0);

  private:
    virtual void setup();
    virtual void run();
    virtual QString validate();
    virtual void loadSettings(const QSettings & settings);
    virtual QString helpURL() {return "Fury";}

  private slots:
    void plotInput(const QString& wsname);
    void rsRangeChangedLazy(double min, double max);
    void updateRS(QtProperty* prop, double val);
    void updatePropertyValues(QtProperty* prop, double val);
    void calculateBinning();
      
  private:
    QwtPlot* m_furPlot;
    MantidWidgets::RangeSelector* m_furRange;
    QwtPlotCurve* m_furCurve;
    QtTreePropertyBrowser* m_furTree;
    QMap<QString, QtProperty*> m_furProp;
    QtDoublePropertyManager* m_furDblMng;
    bool m_furyResFileType;

  };
} // namespace IDA
} // namespace CustomInterfaces
} // namespace MantidQt

#endif /* MANTIDQTCUSTOMINTERFACESIDA_FURY_H_ */
