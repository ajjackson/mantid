//----------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------
#include "MantidAlgorithms/CylinderAbsorption.h"
#include "MantidKernel/PhysicalConstants.h"
#include "MantidAPI/WorkspaceValidators.h"
#include "MantidGeometry/Cylinder.h"
#include "MantidGeometry/Plane.h"
#include "MantidKernel/Fast_Exponential.h"

namespace Mantid
{
namespace Algorithms
{

// Register the algorithm into the AlgorithmFactory
DECLARE_ALGORITHM(CylinderAbsorption)

using namespace Kernel;
using namespace Geometry;
using namespace API;

// Get a reference to the logger. It is used to print out information, warning and error messages
Logger& CylinderAbsorption::g_log = Logger::get("CylinderAbsorption");

CylinderAbsorption::CylinderAbsorption() :
  API::Algorithm(), m_cylinderSample(), m_cylHeight(0.0), m_cylRadius(0.0), m_refAtten(0.0),
      m_scattering(0), m_L1s(), m_Ltot(), m_elementVolumes(), n_lambda(0), x_step(0), exp_options()
{
}

void CylinderAbsorption::init()
{
  declareProperty(new WorkspaceProperty<> ("InputWorkspace", "", Direction::Input,
      new WorkspaceUnitValidator<> ("Wavelength")));
  declareProperty(new WorkspaceProperty<> ("OutputWorkspace", "", Direction::Output));

  BoundedValidator<double> *mustBePositive = new BoundedValidator<double> ();
  mustBePositive->setLower(0.0);
  declareProperty("CylinderSampleHeight", -1.0, mustBePositive,
      "The height of the cylindrical sample in centimetres.");
  declareProperty("CylinderSampleRadius", -1.0, mustBePositive->clone(),
      "The radius of the cylindrical sample in centimetres.");
  declareProperty("AttenuationXSection", -1.0, mustBePositive->clone(),
      "The attenuation cross-section for the sample material in barns.");
  declareProperty("ScatteringXSection", -1.0, mustBePositive->clone(),
      "The scattering cross-section for the sample material in barns.");
  declareProperty("SampleNumberDensity", -1.0, mustBePositive->clone(),
      "The number density of the sample in number per cubic angstrom.");
  BoundedValidator<int> *positiveInt = new BoundedValidator<int> ();
  positiveInt->setLower(1);
  declareProperty("NumberOfSlices", 1, positiveInt,
      "The number of slices into which the cylinder is divided for the calculation.");
  declareProperty("NumberOfAnnuli", 1, positiveInt->clone(),
      "The number of annuli into which each slice is divided for the calculation.");
  declareProperty("NumberOfWavelengthPoints", 1, positiveInt->clone());

  exp_options.push_back("Normal");
  exp_options.push_back("FastApprox");
  declareProperty("ExpMethod", "Normal", new ListValidator(exp_options));
}

void CylinderAbsorption::exec()
{
  // Retrieve the input workspace
  MatrixWorkspace_const_sptr inputWS = getProperty("InputWorkspace");

  // Get the input parameters
  retrieveProperties();
  const double cylinder_volume = m_cylHeight * M_PI * m_cylRadius * m_cylRadius;
  constructCylinderSample();

  // Create the output workspace
  MatrixWorkspace_sptr correctionFactors = WorkspaceFactory::Instance().create(inputWS);
  correctionFactors->setYUnit("Attenuation factor");

  const int numHists = inputWS->getNumberHistograms();
  const int specSize = inputWS->blocksize();

  // If the number of wavelength points has not been given, use them all
  if (n_lambda == 0)
    n_lambda = specSize;
  x_step = specSize / n_lambda; // Bin step between points to calculate

  if (x_step == 0) //Number of wavelength points >number of histogram points
    x_step = 1;

  std::ostringstream message;
  message << "Numerical integration performed every " << x_step << " wavelength points" << std::endl;
  g_log.information(message.str());
  message.str("");

  const bool isHist = inputWS->isHistogramData();
  int iprogress_step = numHists / 100;
  if (iprogress_step == 0)
    iprogress_step = 1;
  // Loop over the spectra
  for (int i = 0; i < numHists; ++i)
  {
    // Copy over bin boundaries
    const MantidVec& X = inputWS->readX(i);
    correctionFactors->dataX(i) = X;

    // Get detector position
    IDetector_const_sptr det;
//    V3D detPos;
    try
    {
      det = inputWS->getDetector(i);
//      detPos = inputWS->getDetector(i)->getPos();
      // Need to convert detPos from metres to cm
//      detPos *= 100.0;
    } catch (Exception::NotFoundError)
    {
      // Catch when a spectrum doesn't have an attached detector and go to next one
      continue;
    }

    this->calculateDistances(det);

    // Get a reference to the Y's in the output WS for storing the factors
    MantidVec& Y = correctionFactors->dataY(i);

    // Loop through the bins in the current spectrum every x_step
    for (int j = 0; j < specSize; j = j + x_step)
    {
      const double lambda = (isHist ? (0.5 * (X[j] + X[j + 1])) : X[j]);
      Y[j] = this->doIntegration(lambda);
      Y[j] /= cylinder_volume; // Divide by total volume of the cylinder
    }

    if (x_step > 1) // Interpolate linearly between points separated by x_step, last point required
    {
      const double lambda = (isHist ? (0.5 * (X[specSize - 1] + X[specSize])) : X[specSize - 1]);
      Y[specSize - 1] = this->doIntegration(lambda);
      Y[specSize - 1] /= cylinder_volume;
      interpolate(X, Y, isHist);
    }

    if (i % iprogress_step == 0)
    {
      progress(double(i) / numHists);
      interruption_point();
    }

  }
  g_log.information() << "Total number of elements in the integration was " << m_L1s.size() << std::endl;
  setProperty("OutputWorkspace", correctionFactors);

  // Now do some cleaning-up since destructor is not called
  m_L1s.clear();
  m_Ltot.clear();
  m_elementVolumes.clear();
  exp_options.clear();
}

/// Fetch the properties and set the appropriate member variables
void CylinderAbsorption::retrieveProperties()
{
  m_cylHeight = getProperty("CylinderSampleHeight"); // in cm
  m_cylRadius = getProperty("CylinderSampleRadius"); // in cm
  const double sigma_atten = getProperty("AttenuationXSection"); // in barns
  const double sigma_s = getProperty("ScatteringXSection"); // in barns
  const double rho = getProperty("SampleNumberDensity"); // in Angstroms-3
  m_refAtten = -sigma_atten * rho / 1.798;
  m_scattering = -sigma_s * rho;

  n_lambda = getProperty("NumberOfWavelengthPoints");

  std::string exp_string = getProperty("ExpMethod");

  if (exp_string == "Normal") // Use the system exp function
    EXPONENTIAL = exp;
  else if (exp_string == "FastApprox") // Use the compact approximation
    EXPONENTIAL = fast_exp;
}

/// Create the cylinder object using the Geometry classes
void CylinderAbsorption::constructCylinderSample()
{
  std::map<int, Surface*> surfaces;

  const V3D normVec(0.0, 1.0, 0.0);

  Cylinder* cyl = new Cylinder();
  // For now, assume beam comes in along z axis, that y is up and that sample is at origin
  cyl->setCentre(V3D(0.0, 0.0, 0.0));
  cyl->setNorm(normVec);
  cyl->setRadius(m_cylRadius);
  surfaces[1] = cyl;

  Plane* top = new Plane();
  V3D pointInPlane = normVec * (m_cylHeight / 2.0);
  top->setPlane(pointInPlane, normVec);
  surfaces[2] = top;
  Plane* bottom = new Plane();
  pointInPlane[1] *= -1.0;
  bottom->setPlane(pointInPlane, normVec);
  surfaces[3] = bottom;

  int success = m_cylinderSample.setObject(21, "-1 -2 3");
  assert(success);
  success = m_cylinderSample.populate(surfaces);
  assert(!success);

  const double radius = m_cylRadius;
  assert(m_cylinderSample.isValid(V3D(0.0, 0.0, 0.0)));
  assert(!m_cylinderSample.isValid(V3D(radius + 0.001, 0.0, 0.0)));
  assert(m_cylinderSample.isValid(V3D(radius - 0.001, 0.0, 0.0)));
  assert(!m_cylinderSample.isValid(V3D(0.0, m_cylHeight, 0.0)));
  assert(!m_cylinderSample.isValid(V3D(0.0, -1.0 * m_cylHeight, 0.0)));
  assert(m_cylinderSample.isOnSide(V3D(radius, 0.0, 0.0)));
  assert(m_cylinderSample.isOnSide(V3D(0.0, m_cylHeight / 2.0, 0.0)));
  assert(m_cylinderSample.isOnSide(V3D(radius, m_cylHeight / 2.0, 0.0)));

  g_log.information("Successfully constructed the sample object");
}

/// Calculate the distances traversed by the neutrons within the sample
void CylinderAbsorption::calculateDistances(const Geometry::IDetector_const_sptr& detector)
{
  // Test whether I need to calculate distances
  const bool calculateL1s = m_L1s.empty();

  const int numSlices = getProperty("NumberOfSlices");
  const double sliceThickness = m_cylHeight / numSlices;
  const int numAnnuli = getProperty("NumberOfAnnuli");
  ;
  const double deltaR = m_cylRadius / numAnnuli;

  /* The number of volume elements is
   * numslices*(1+2+3+.....+numAnnuli)*6
   * Since the first annulus is separated in 6 segments, the next one in 12 and so on.....
   */
  if (calculateL1s)
  {
    int n_volume_elements = numSlices * numAnnuli * (numAnnuli + 1) * 3;

    m_L1s.resize(n_volume_elements);
    m_Ltot.resize(n_volume_elements);
    m_elementVolumes.resize(n_volume_elements);
  }
  int counter = 0;
  // loop over slices

  V3D currentPosition;

  for (int i = 0; i < numSlices; ++i)
  {
    const double z = (i + 0.5) * sliceThickness - 0.5 * m_cylHeight;

    // Number of elements in 1st annulus
    int Ni = 0;
    // loop over annuli
    for (int j = 0; j < numAnnuli; ++j)
    {
      Ni += 6;
      const double R = (j * m_cylRadius / numAnnuli) + (deltaR / 2.0);
      // loop over elements in current annulus
      for (int k = 0; k < Ni; ++k)
      {
        // These need only be calculated once
        if (calculateL1s)
        {
          const double phi = 2* M_PI * k / Ni;
          // Calculate the current position in the sample in Cartesian coordinates.
          // Remember that our cylinder has its axis along the y axis
          currentPosition(R * sin(phi), z, R * cos(phi));
          assert(m_cylinderSample.isValid(currentPosition));
          // Create track for distance in cylinder before scattering point
          // Remember beam along Z direction
          Track incoming(currentPosition, V3D(0.0, 0.0, -1.0));
          int tmp = m_cylinderSample.interceptSurface(incoming);
          assert(tmp == 1);
          m_L1s[counter] = incoming.begin()->Dist;

          // Also calculate element volumes here
          const double outerR = R + (deltaR / 2.0);
          const double innerR = outerR - deltaR;
          const double elementVolume = M_PI * (outerR * outerR - innerR * innerR) * sliceThickness / Ni;
          m_elementVolumes[counter] = elementVolume;
        }

        // We need to make sure this is right for grouped detectors - should use average theta & phi
        V3D detectorPos;
        detectorPos.spherical(100.0*detector->getDistance(Component("dummy",currentPosition)),
            detector->getTwoTheta(currentPosition,V3D(0.0,0.0,1.0)),detector->getPhi());
        // Create track for distance in cylinder between scattering point and detector
        V3D direction = detectorPos - currentPosition;
        direction.normalize();
        Track outgoing(currentPosition, direction);
        int temp = m_cylinderSample.interceptSurface(outgoing);

        /* Most of the time, the number of hits is 1. Sometime, we have more than one intersection due to
         * arithmetic imprecision. If it is the case, then selecting the first intersection is valid.
         * In principle, one could check the consistency of all distances if hits is larger than one by doing:
         * Mantid::Geometry::Track::LType::const_iterator it=outgoing.begin();
         * and looping until outgoing.end() checking the distances with it->Dist
         */
        // Not hitting the cylinder from inside, usually means detector is badly defined,
        // i.e, position is (0,0,0).
        if (temp < 1)
        {
          std::ostringstream message;
          message << "Problem with detector at" << detectorPos << std::endl;
          message << "This usually means that this detector is defined inside the sample cylinder";
          g_log.error(message.str());
          throw std::runtime_error("Problem in CylinderAbsorption::calculateDistances");
        }
        m_Ltot[counter] = outgoing.begin()->Dist + m_L1s[counter];
        counter++;
      }
    }
  }
}

/// Carries out the numerical integration over the cylinder
double CylinderAbsorption::doIntegration(const double& lambda) const
{
  double integral = 0.0;
  double exponent;

  int el = m_Ltot.size();

  // Iterate over all the elements, summing up the integral
  for (int i = 0; i < el; ++i)
  {
    // Equation is exponent * element volume
    // where exponent is e^(-mu * wavelength/1.8 * (L1+L2) )  (N.B. distances are in cm)
    exponent = ((m_refAtten * lambda) + m_scattering) * (m_Ltot[i]);
    integral += (EXPONENTIAL(exponent) * (m_elementVolumes[i]));
  }

  return integral;
}

/// Calculate the value of the correction at the points not explicitly numerically integrated
void CylinderAbsorption::interpolate(const MantidVec& x, MantidVec& y, bool is_histogram)
{
  int step = x_step, index2;
  double x1 = 0, x2 = 0, y1 = 0, y2 = 0, xp, overgap = 0;
  int specsize = y.size();
  for (int i = 0; i < specsize - 1; ++i) // Last point has been calculated
  {
    if (step == x_step) //Point numerically integrated, does not need interpolation
    {
      x1 = (is_histogram ? (0.5 * (x[i] + x[i + 1])) : x[i]);
      index2 = ((i + x_step) >= specsize ? specsize - 1 : (i + x_step));
      x2 = (is_histogram ? (0.5 * (x[index2] + x[index2 + 1])) : x[index2]);
      overgap = 1.0 / (x2 - x1);
      y1 = y[i];
      y2 = y[index2];
      step = 1;
      continue;
    }
    xp = (is_histogram ? (0.5 * (x[i] + x[i + 1])) : x[i]);
    // Interpolation
    y[i] = (xp - x1) * y2 + (x2 - xp) * y1;
    y[i] *= overgap;
    step++;
  }
  return;
}

} // namespace Algorithms
} // namespace Mantid
